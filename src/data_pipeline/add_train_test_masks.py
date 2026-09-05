import pickle
import numpy as np
import torch

EXPORT_PATH = "data/graph_export.pkl"
HETERO_PATH = "data/hetero_data.pt"


def add_masks():
    with open(EXPORT_PATH, "rb") as f:
        export = pickle.load(f)

    data = torch.load(HETERO_PATH, weights_only=False)

   
    dt_values = np.array([t["dt"] for t in export["nodes"]["transaction"]])

    n = len(dt_values)
    sorted_order = np.argsort(dt_values)  
    split_point = int(n * 0.8)

    train_positions = sorted_order[:split_point]
    test_positions = sorted_order[split_point:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_positions] = True
    test_mask[test_positions] = True

    data["transaction"].train_mask = train_mask
    data["transaction"].test_mask = test_mask

    print(f"Train: {train_mask.sum().item()} transactions")
    print(f"Test:  {test_mask.sum().item()} transactions")

    y = data["transaction"].y
    print(f"Train fraud rate: {y[train_mask].float().mean().item():.4f}")
    print(f"Test fraud rate:  {y[test_mask].float().mean().item():.4f}")

    torch.save(data, HETERO_PATH)
    print(f"\nUpdated {HETERO_PATH} with train_mask/test_mask")
    return data


if __name__ == "__main__":
    add_masks()