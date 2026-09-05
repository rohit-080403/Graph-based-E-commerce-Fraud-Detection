import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class FraudGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2):
        super().__init__()

        self.conv1 = HeteroConv({
            ("transaction", "used_card", "card"): SAGEConv((-1, -1), hidden_dim),
            ("card", "rev_used_card", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "from_address", "address"): SAGEConv((-1, -1), hidden_dim),
            ("address", "rev_from_address", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "used_device", "device"): SAGEConv((-1, -1), hidden_dim),
            ("device", "rev_used_device", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "used_email", "email"): SAGEConv((-1, -1), hidden_dim),
            ("email", "rev_used_email", "transaction"): SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")  

        self.conv2 = HeteroConv({
            ("transaction", "used_card", "card"): SAGEConv((-1, -1), hidden_dim),
            ("card", "rev_used_card", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "from_address", "address"): SAGEConv((-1, -1), hidden_dim),
            ("address", "rev_from_address", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "used_device", "device"): SAGEConv((-1, -1), hidden_dim),
            ("device", "rev_used_device", "transaction"): SAGEConv((-1, -1), hidden_dim),
            ("transaction", "used_email", "email"): SAGEConv((-1, -1), hidden_dim),
            ("email", "rev_used_email", "transaction"): SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

       
        out = self.classifier(x_dict["transaction"])
        return out