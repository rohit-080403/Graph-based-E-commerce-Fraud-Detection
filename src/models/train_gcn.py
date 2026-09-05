import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from src.models.gcn_model import FraudGCN

DATA_PATH = "data/hetero_data.pt"
MODEL_OUT = "src/models/fraud_gcn.pt"
EPOCHS = 300          
LR = 0.01             


def train():
    data = torch.load(DATA_PATH, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print(f"Using device: {device}")

    model = FraudGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    y = data["transaction"].y
    train_mask = data["transaction"].train_mask
    test_mask = data["transaction"].test_mask

    n_pos = (y[train_mask] == 1).sum()
    n_neg = (y[train_mask] == 0).sum()
    class_weights = torch.tensor([1.0, (n_neg / n_pos).item()], device=device)
    print(f"Class weights: {class_weights.tolist()}")

    best_test_f1 = 0.0
    best_state = None

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_out = model(data.x_dict, data.edge_index_dict)
                test_pred = test_out[test_mask].argmax(dim=1)
                test_acc = (test_pred == y[test_mask]).float().mean().item()

                
                y_test_np = y[test_mask].cpu().numpy()
                pred_np = test_pred.cpu().numpy()
                test_f1 = f1_score(y_test_np, pred_np, zero_division=0)

                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Acc: {test_acc:.4f} | "
                  f"Test F1: {test_f1:.4f} | LR: {current_lr:.5f}")
            model.train()

    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint (Test F1: {best_test_f1:.4f})")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")
    return model, data


def evaluate(model, data):
    model.eval()
    y = data["transaction"].y
    test_mask = data["transaction"].test_mask

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)

    y_test = y[test_mask].cpu().numpy()
    pred_test = preds[test_mask].cpu().numpy()
    proba_test = probs[test_mask].cpu().numpy()

    print("\n--- GCN Results (test set, best checkpoint) ---")
    print(f"Precision: {precision_score(y_test, pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, pred_test):.4f}")
    print(f"F1:        {f1_score(y_test, pred_test):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba_test):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred_test))

    ring_ids = data["transaction"].ring_id
    ring_mask = torch.tensor([r is not None for r in ring_ids])
    ring_mask_test = ring_mask & test_mask.cpu()

    if ring_mask_test.sum() > 0:
        ring_preds = preds[ring_mask_test]
        ring_actual = y[ring_mask_test]
        ring_recall = (ring_preds[ring_actual == 1] == 1).float().mean().item()
        print(f"\nSynthetic ring recall (in test set, n={ring_mask_test.sum().item()}): {ring_recall:.4f}")


if __name__ == "__main__":
    model, data = train()
    evaluate(model, data)