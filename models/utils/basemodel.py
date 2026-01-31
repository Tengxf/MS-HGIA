import torch
import torch.nn as nn
from losses import *
from utils import dropout_node


def ContrastiveCritLoss(feats_lower, feats_higher, childof, temperature=0.1):
    feats_lower = F.normalize(feats_lower, p=2, dim=1)
    feats_higher = F.normalize(feats_higher, p=2, dim=1)

    valid_mask = (childof >= 0) & (childof < feats_lower.size(0))
    feats_higher = feats_higher[valid_mask]
    childof = childof[valid_mask]

    if feats_higher.numel() == 0:
        return torch.tensor(0.0, device=feats_lower.device)

    unique_parents, parent_inverse = torch.unique(childof, sorted=True, return_inverse=True)
    active_lower = feats_lower[unique_parents]

    sim_matrix = torch.matmul(active_lower, feats_higher.T) / temperature
    target = parent_inverse
    sim_matrix = sim_matrix - sim_matrix.max(dim=0, keepdim=True)[0].detach()

    loss = F.cross_entropy(sim_matrix.T, target)

    return loss

def CrossScaleConsistencyLoss(feats_low, feats_high, childof, mode='l2', tau=0.1):
    valid_mask = (childof >= 0) & (childof < feats_low.size(0))
    feats_high = feats_high[valid_mask]
    childof = childof[valid_mask]
    if feats_high.numel() == 0:
        return torch.tensor(0.0, device=feats_low.device)

    parent_feats = feats_low[childof]

    if mode == 'l2':
        loss = F.mse_loss(F.normalize(feats_high, dim=1), F.normalize(parent_feats, dim=1))
    elif mode == 'cosine':
        feats_high = F.normalize(feats_high, p=2, dim=1)
        parent_feats = F.normalize(parent_feats, p=2, dim=1)
        sim = F.cosine_similarity(feats_high, parent_feats, dim=1)
        loss = (1 - sim).mean()
    elif mode == 'kl':
        p = F.log_softmax(feats_high / tau, dim=1)
        q = F.softmax(parent_feats / tau, dim=1)
        loss = F.kl_div(p, q, reduction="batchmean")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return loss

class Baseline(nn.Module):

    def __init__(self, args, state_dict_weights=None):
        super().__init__()
        self.args=args
        self.target= args.target
        self.lamb=args.lamb
        self.beta=args.beta
        self.dropout= args.dropout
        self.tau=args.temperature
        self.kl= args.kl
        self.residual=args.residual
        self.c_in=args.input_size
        self.classes=args.n_classes
        self.c_hidden=args.c_hidden
        self.add_bias=args.add_bias
        self.max=args.max
        self.state_dict_weights=state_dict_weights
        self.contrastive_temp = args.contrastive_temp
        self.consistency_weight = args.consistency_weight
        self.contrastive_weight = args.contrastive_weight


    def forward_scale(self, x, edge_index, gnnlayer, glu=None):
        r = x
        if self.training and self.args.dropout:
            edge_index, _, _ = dropout_node(edge_index=edge_index)
        x = gnnlayer(x, edge_index=edge_index, train=self.training)
        if self.residual:
            if glu is not None:
                x=glu(x)
            x=r+x
        else:
            x = x
        return x, edge_index

    def forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        NotImplementedError("forward_gnn error")

    def forward_mil(self, indecesperlevel: torch.Tensor, feats: torch.Tensor, results: dict):
        NotImplementedError("forward_mil error")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        feats, indecesperlevel, results = self.forward_gnn(
            x, edge_index, levels, childof, edge_index2, edge_index3)
        results = self.forward_mil(indecesperlevel, feats, results)
        return results

    def compute_loss(self, loss_module_instance: torch.nn.Module, results: dict, bag_label: torch.Tensor, epoch) -> torch.Tensor:

        loss = 0
        a = 0.5
        WARMUP_EPOCHS = 15.0
        warmup_factor = min(1.0, epoch / WARMUP_EPOCHS)

        y_instance_pred1, prediction_bag1, _, _ = results[self.target]

        if self.classes > 2:
            if prediction_bag1.dim() == 1:
                prediction_bag1 = prediction_bag1.unsqueeze(0)
            bag_label_ce = bag_label.view(-1).long()

            higher_loss = loss_module_instance(prediction_bag1, bag_label_ce)
        else:
            higher_loss = CELoss(loss_module_instance,y_instance_pred1, prediction_bag1, bag_label)

        loss += higher_loss

        if self.kl is not None:
            y_instance_pred2, prediction_bag2, _, _ = results[self.kl]
            if self.classes > 2:
                if prediction_bag2.dim() == 1:
                    prediction_bag2 = prediction_bag2.unsqueeze(0)
                bag_label_ce = bag_label.view(-1).long()
                lower_loss = loss_module_instance(prediction_bag2, bag_label_ce)
            else:
                lower_loss = CELoss(loss_module_instance, y_instance_pred2, prediction_bag2, bag_label,
                                    lamb=(1.0 - self.lamb))
            loss += lower_loss

            current_contrastive_weight = 0.0
            if "gnn_feats_lv0" in results and "gnn_feats_lv1" in results:
                contrastive_crit_loss = ContrastiveCritLoss(
                    feats_lower=results["gnn_feats_lv0"].squeeze(0),
                    feats_higher=results["gnn_feats_lv1"].squeeze(0),
                    childof=results["childof"],
                    temperature=self.contrastive_temp
                )
                current_contrastive_weight = warmup_factor * self.args.contrastive_weight
                loss += current_contrastive_weight * contrastive_crit_loss
            else:
                print("[WARN] Missing gnn_feats_lv0 or gnn_feats_lv1, skip contrastive loss.")

            current_consistency_weight = 0.0
            if "gnn_feats_lv0" in results and "fused_high" in results:
                consistency_loss = CrossScaleConsistencyLoss(
                    feats_low=results["gnn_feats_lv0"].squeeze(0),
                    feats_high=results["fused_high"].squeeze(0),
                    childof=results["childof"],
                    mode='cosine'
                )
                current_consistency_weight = warmup_factor * self.consistency_weight
                loss += current_consistency_weight * consistency_loss
            else:
                print("[WARN] Missing fused_high or gnn_feats_lv0, skip consistency loss.")

            current_lamb_weight = 0.0
            kl_bag_loss = computeKL(
                prediction_bag2, prediction_bag1, self.tau, self.classes, self.add_bias)
            current_lamb_weight = warmup_factor * self.args.gamma
            loss += current_lamb_weight * kl_bag_loss

        return loss

    def predict(self, results):

        prediction_patch_higher, prediction_bag_higher, _, _ = results[self.target]

        if self.classes > 1:
            higher_prediction = torch.softmax(prediction_bag_higher, dim=1)
            if self.kl is not None:
                _, prediction_bag_lower, _, _ = results[self.kl]
                lower_prediction = torch.softmax(prediction_bag_lower, dim=1)
            else:
                lower_prediction = None
        else:
            higher_prediction = 0.5 * \
                torch.sigmoid(prediction_bag_higher)[
                    :, -1]+0.5 * torch.sigmoid(prediction_bag_higher)[:, -1]
            if self.kl is not None:
                _, prediction_bag_lower, _, _ = results[self.kl]
                lower_prediction = torch.sigmoid(prediction_bag_lower)[:, -1]
            else:
                lower_prediction = None
        return higher_prediction, lower_prediction
