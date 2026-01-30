import torch
from torch_scatter import scatter_add

def get_sparse_hypergraph_knn_and_threshold(x: torch.Tensor,
                                            k_nearest: int,
                                            k_threshold: float,
                                            largest: bool = False,
                                            normalize_dist: bool = True):

    V, C = x.shape
    device = x.device
    x_float = x.float()

    with torch.no_grad():
        dis_matrix = torch.cdist(x_float, x_float, p=2)

        norm_dis_matrix = dis_matrix
        if normalize_dist:
            max_dis, _ = dis_matrix.max(0, keepdim=True)
            norm_dis_matrix = dis_matrix / (max_dis + 1e-6)
            del max_dis

        k_nearest = min(k_nearest, V)

    knn_dist, knn_idx = torch.topk(norm_dis_matrix, k_nearest, dim=0, largest=largest)

    node_idx_knn = knn_idx.t().reshape(-1)
    edge_idx_knn = torch.arange(V, device=device).repeat_interleave(k_nearest)
    edge_index_knn = torch.stack([node_idx_knn, edge_idx_knn], dim=0)

    avg_dist_knn = knn_dist.mean(dim=0)
    hyedge_weight_knn = (1.0 - avg_dist_knn).clamp(0.0, 1.0)
    del knn_dist, knn_idx

    node_idx_thresh, edge_idx_thresh = torch.where(norm_dis_matrix < k_threshold)

    edge_index_thresh = None
    hyedge_weight_thresh = torch.zeros(V, device=device, dtype=hyedge_weight_knn.dtype)

    if node_idx_thresh.numel() > 0:
        edge_idx_thresh_offset = edge_idx_thresh + V
        edge_index_thresh = torch.stack([node_idx_thresh, edge_idx_thresh_offset], dim=0)

        dists_thresh = norm_dis_matrix[node_idx_thresh, edge_idx_thresh]

        summed_dists_per_edge = torch.zeros(V, device=device, dtype=torch.float32)
        scatter_add(dists_thresh.float(), edge_idx_thresh, dim=0, out=summed_dists_per_edge)

        counts_per_edge = torch.zeros(V, device=device, dtype=torch.float32)
        scatter_add(torch.ones_like(edge_idx_thresh, dtype=torch.float32), edge_idx_thresh, dim=0,
                    out=counts_per_edge)

        avg_dist_thresh = summed_dists_per_edge / (counts_per_edge + 1e-6)

        hyedge_weight_thresh = (1.0 - avg_dist_thresh).clamp(0.0, 1.0)

    del norm_dis_matrix, dis_matrix, node_idx_thresh, edge_idx_thresh

    if edge_index_thresh is not None:
        final_edge_index = torch.cat([edge_index_knn, edge_index_thresh], dim=1)
    else:
        final_edge_index = edge_index_knn

    final_hyedge_weight = torch.cat([
        hyedge_weight_knn.to(x.dtype),
        hyedge_weight_thresh.to(x.dtype)
    ], dim=0)

    num_hyperedges = 2 * V

    return final_edge_index, final_hyedge_weight, num_hyperedges
