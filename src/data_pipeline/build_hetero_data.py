import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, LabelEncoder

EXPORT_PATH = "data/graph_export.pkl"
SAMPLE_PATH = "data/processed_sample.parquet"
OUT_PATH = "data/hetero_data.pt"

EXCLUDE_COLS = [
    "TransactionID", "isFraud",
    "card_id", "address_id", "device_id", "email_domain"
]


def load_export():
    with open(EXPORT_PATH, "rb") as f:
        return pickle.load(f)


def build_transaction_features(export):
    df = pd.read_parquet(SAMPLE_PATH)

    txn_order = [t["node_id"] for t in export["nodes"]["transaction"]]
    df = df.set_index("TransactionID").loc[txn_order].reset_index()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df[feature_cols] = df[feature_cols].fillna(0)

    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    x = torch.tensor(scaled, dtype=torch.float)
    y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    ring_ids = [t["ring_id"] for t in export["nodes"]["transaction"]]

    print(f"Transaction features: {x.shape} (standardized)")
    return x, y, ring_ids


def build_structural_features(nodes, index_map, edge_list):
    n = len(nodes)
    degree = np.zeros(n)

    for src, tgt in edge_list:
        tgt_idx = index_map[tgt]
        degree[tgt_idx] += 1


    degree_log = np.log1p(degree)

    is_synthetic = np.zeros(n)
    for i, node in enumerate(nodes):
        if node.get("is_synthetic"):
            is_synthetic[i] = 1

    scaler = StandardScaler()
    degree_scaled = scaler.fit_transform(degree_log.reshape(-1, 1)).flatten()

    x = np.stack([degree_scaled, is_synthetic], axis=1)
    x = torch.tensor(x, dtype=torch.float)
    print(f"Node features: {x.shape} (log-scaled + standardized)")
    return x


def build_edge_index(edge_list, src_index_map, tgt_index_map):
    src_idx = [src_index_map[s] for s, t in edge_list]
    tgt_idx = [tgt_index_map[t] for s, t in edge_list]
    return torch.tensor([src_idx, tgt_idx], dtype=torch.long)


def build_hetero_data():
    export = load_export()
    data = HeteroData()

    x_txn, y_txn, ring_ids = build_transaction_features(export)
    data["transaction"].x = x_txn
    data["transaction"].y = y_txn
    data["transaction"].ring_id = ring_ids

    data["card"].x = build_structural_features(
        export["nodes"]["card"], export["index_maps"]["card"], export["edges"]["txn_card"])
    data["address"].x = build_structural_features(
        export["nodes"]["address"], export["index_maps"]["address"], export["edges"]["txn_address"])
    data["device"].x = build_structural_features(
        export["nodes"]["device"], export["index_maps"]["device"], export["edges"]["txn_device"])
    data["email"].x = build_structural_features(
        export["nodes"]["email"], export["index_maps"]["email"], export["edges"]["txn_email"])

    txn_idx = export["index_maps"]["transaction"]

    data["transaction", "used_card", "card"].edge_index = build_edge_index(
        export["edges"]["txn_card"], txn_idx, export["index_maps"]["card"])
    data["card", "rev_used_card", "transaction"].edge_index = data[
        "transaction", "used_card", "card"].edge_index.flip(0)

    data["transaction", "from_address", "address"].edge_index = build_edge_index(
        export["edges"]["txn_address"], txn_idx, export["index_maps"]["address"])
    data["address", "rev_from_address", "transaction"].edge_index = data[
        "transaction", "from_address", "address"].edge_index.flip(0)

    data["transaction", "used_device", "device"].edge_index = build_edge_index(
        export["edges"]["txn_device"], txn_idx, export["index_maps"]["device"])
    data["device", "rev_used_device", "transaction"].edge_index = data[
        "transaction", "used_device", "device"].edge_index.flip(0)

    data["transaction", "used_email", "email"].edge_index = build_edge_index(
        export["edges"]["txn_email"], txn_idx, export["index_maps"]["email"])
    data["email", "rev_used_email", "transaction"].edge_index = data[
        "transaction", "used_email", "email"].edge_index.flip(0)

    print("\n--- HeteroData summary ---")
    print(data)

    torch.save(data, OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")
    return data


if __name__ == "__main__":
    build_hetero_data()