import torch
import torch.nn as nn
from models.utils.hypergraph_layer import HyConv
from models.utils.hypergraph_build import *
from torch_scatter import scatter_add

class HyperGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=1,
                 dp_rate=0.3, drop_max_ratio=0.2, k_nearest=4, build_hypergraph: bool = True, apply_head=True):
        super(HyperGNNModel, self).__init__()
        self.k_nearest = k_nearest
        self.build_hypergraph = build_hypergraph
        self.layers = nn.ModuleList()
        self.layers.append(HyConv(c_in, c_hidden, dropout=dp_rate, drop_max_ratio=drop_max_ratio))
        for _ in range(num_layers - 1):
            self.layers.append(HyConv(c_hidden, c_hidden, dropout=dp_rate, drop_max_ratio=drop_max_ratio))
        self.linear_out = nn.Linear(c_hidden, c_out) if apply_head else None
        self.dropout = nn.Dropout(dp_rate) if apply_head else nn.Identity()

    def forward(self, x, edge_index=None, train=True):
        hyedge_weight = None
        num_hyperedges = None

        if x.ndim == 3:
            assert x.shape[0] == 1, "Sparse mode only supports batch size B=1"
            x = x.squeeze(0)

        if self.build_hypergraph:
            edge_index, hyedge_weight, num_hyperedges = get_sparse_hypergraph_knn_and_threshold(
                x,
                k_nearest=self.k_nearest,
                k_threshold=0.1,
                largest=False
            )

        else:
            assert edge_index is not None, "edge_index must be provided in static graph mode"
            if edge_index.ndim == 3 and edge_index.shape[0] == 1:
                edge_index = edge_index.squeeze(0)
            assert edge_index.ndim == 2 and edge_index.shape[0] == 2, \
                f"Static edge_index should be [2, M], but got {list(edge_index.shape)}"
            num_hyperedges = edge_index[1].max().item() + 1
            hyedge_weight = None

        for conv in self.layers:
            x = conv(x, edge_index,
                     hyedge_weight=hyedge_weight,
                     num_hyperedges=num_hyperedges,
                     train=train)

        if train:
            x = self.dropout(x)
        if self.linear_out is not None:
            x = self.linear_out(x)

        return x.unsqueeze(0)