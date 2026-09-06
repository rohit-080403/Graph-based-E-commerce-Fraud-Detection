import torch
from torch_geometric.explain import Explainer, GNNExplainer

from src.models.gcn_model import FraudGCN

DATA_PATH = "data/hetero_data.pt"
MODEL_PATH = "src/models/fraud_gcn.pt"


def load_model_and_data():
    data = torch.load(DATA_PATH, weights_only=False)
    model = FraudGCN()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model, data


def extract_khop_subgraph(data, txn_index):
    
    from torch_geometric.data import HeteroData

    sub = HeteroData()
    entity_types = ["card", "device"]
    rel_map = {
        "card": "used_card", "address": "from_address",
        "device": "used_device", "email": "used_email",
    }

    connected_entities = {}
    for etype in entity_types:
        edge_index = data["transaction", rel_map[etype], etype].edge_index
        mask = edge_index[0] == txn_index
        entity_ids = edge_index[1][mask]
        connected_entities[etype] = entity_ids

    all_txn_indices = {txn_index}
    for etype in entity_types:
        rev_edge_index = data[etype, f"rev_{rel_map[etype]}", "transaction"].edge_index
        for eid in connected_entities[etype]:
            mask = rev_edge_index[0] == eid
            neighbor_txns = rev_edge_index[1][mask]
            all_txn_indices.update(neighbor_txns.tolist())

    all_txn_indices = sorted(all_txn_indices)
    print(f"Subgraph size: {len(all_txn_indices)} transactions "
          f"(vs {data['transaction'].x.shape[0]} in full graph)")

    txn_remap = {old: new for new, old in enumerate(all_txn_indices)}
    sub["transaction"].x = data["transaction"].x[all_txn_indices]
    sub["transaction"].y = data["transaction"].y[all_txn_indices]
    new_txn_index = txn_remap[txn_index]

    for etype in entity_types:
        entity_ids = sorted(set(connected_entities[etype].tolist()))
        entity_remap = {old: new for new, old in enumerate(entity_ids)}
        sub[etype].x = data[etype].x[entity_ids]

        edge_index = data["transaction", rel_map[etype], etype].edge_index
        src, tgt = [], []
        for i in range(edge_index.shape[1]):
            s, t = edge_index[0, i].item(), edge_index[1, i].item()
            if s in txn_remap and t in entity_remap:
                src.append(txn_remap[s])
                tgt.append(entity_remap[t])
        sub["transaction", rel_map[etype], etype].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        sub[etype, f"rev_{rel_map[etype]}", "transaction"].edge_index = torch.tensor([tgt, src], dtype=torch.long)

    return sub, new_txn_index


def build_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),  
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def summarize_explanation(explanation, sub, txn_index):
    print(f"\n--- Explanation for transaction (local index {txn_index}) ---")
    actual_label = sub["transaction"].y[txn_index].item()
    print(f"Actual label: {'FRAUD' if actual_label == 1 else 'CLEAN'}")

    print("\nEdge importance by relationship type:")
    for edge_type, mask in explanation.edge_mask_dict.items():
        if mask.numel() > 0:
            print(f"  {edge_type}: mean = {mask.mean().item():.4f}, max = {mask.max().item():.4f}")

    print("\nTop contributing transaction features:")
    txn_feat_mask = explanation.node_mask_dict["transaction"][txn_index]
    top_features = torch.topk(txn_feat_mask, k=10)
    for idx, val in zip(top_features.indices.tolist(), top_features.values.tolist()):
        print(f"  Feature index {idx}: importance = {val:.4f}")


if __name__ == "__main__":
    model, data = load_model_and_data()
    explainer = build_explainer(model)

    test_mask = data["transaction"].test_mask
    ring_ids = data["transaction"].ring_id
    ring_indices = [i for i, r in enumerate(ring_ids) if r is not None and test_mask[i]]

    if not ring_indices:
        print("No ring transactions in test set — falling back to first fraud transaction")
        y = data["transaction"].y
        ring_indices = (test_mask & (y == 1)).nonzero(as_tuple=True)[0].tolist()

    target_idx = ring_indices[0]
    print(f"Explaining transaction at global index {target_idx}")

    sub, local_idx = extract_khop_subgraph(data, target_idx)
    explanation = explainer(x=sub.x_dict, edge_index=sub.edge_index_dict, index=local_idx)
    summarize_explanation(explanation, sub, local_idx)