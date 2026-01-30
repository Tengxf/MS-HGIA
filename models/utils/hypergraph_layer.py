import torch
import torch.nn as nn
from torch import einsum
from torch.nn import Parameter
from torch.cuda.amp import autocast
from torch_scatter import scatter_add


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3, drop_max_ratio=0.2, bias=True):
        super().__init__()
        self.drop_out = dropout
        self.drop_max_ratio = drop_max_ratio

        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def dropmax(self, x, drop_ratio):
        N, V, C = x.size()
        drop_num = int(V * drop_ratio)
        max_topk, _ = torch.topk(x, drop_num, dim=1)
        max_threshold = max_topk[:, -1, :].unsqueeze(1)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask = torch.where(x <= max_threshold, zeros, ones)
        mask = (mask * torch.gt(torch.rand_like(mask), 1 - self.drop_out)) * -1 + 1
        x = x * mask
        return x

    def forward(self, x, edge_index, hyedge_weight=None, train=True,
                coors=None, num_hyperedges=None):

        V, C_in = x.shape

        if num_hyperedges is None:
            E = edge_index[1].max().item() + 1
        else:
            E = num_hyperedges

        node_idx, edge_idx = edge_index[0], edge_index[1]  # M
        epsilon = 1e-6

        with autocast(dtype=torch.float16):
            y = torch.matmul(x, self.theta)

        De_inv = None
        Dv_inv = None

        with autocast(enabled=False):
            y_float = y.float()

            De = scatter_add(torch.ones_like(node_idx, dtype=torch.float32),
                             edge_idx, dim=0, dim_size=E)
            De_inv = (1.0 / (De + epsilon)).unsqueeze(1)

            if hyedge_weight is not None:
                w_de = hyedge_weight.float()[edge_idx]
                Dv = scatter_add(w_de, node_idx, dim=0, dim_size=V)
            else:
                Dv = scatter_add(torch.ones_like(edge_idx, dtype=torch.float32),
                                 node_idx, dim=0, dim_size=V)
            Dv_inv = (1.0 / (Dv + epsilon)).unsqueeze(1)  # [V, 1]

        y_edge_feats = scatter_add(y_float[node_idx], edge_idx, dim=0, dim_size=E)

        y_edge_feats = y_edge_feats * De_inv

        if hyedge_weight is not None:
            y_edge_feats = y_edge_feats * hyedge_weight.float().unsqueeze(1)

        if train:
            y_edge_feats_3d = y_edge_feats.unsqueeze(0)
            y_edge_feats_3d = self.dropmax(y_edge_feats_3d, self.drop_max_ratio)
            y_edge_feats = y_edge_feats_3d.squeeze(0)

        y = scatter_add(y_edge_feats[edge_idx], node_idx, dim=0, dim_size=V)
        y = y * Dv_inv

        if self.bias is not None:
            y = y + self.bias

        y = y.to(dtype=self.theta.dtype)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.relu(y)