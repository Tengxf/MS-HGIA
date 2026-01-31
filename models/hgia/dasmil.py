from models.utils.basemodel import Baseline
from models.utils.modules import FCLayer, BClassifier, MILNet, init, GatedLinearUnit
from models.utils.hypergnn_model import HyperGNNModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class ChildGuidedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        # self.norm_v = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, feats_high, feats_low, childof=None):

        q = self.q_proj(self.norm_q(feats_high)).unsqueeze(0)
        k = self.k_proj(self.norm_k(feats_low)).unsqueeze(0)
        # v = self.v_proj(feats_low.unsqueeze(0))
        v_float32 = None
        with autocast(enabled=False):
            v_float32 = self.v_proj(feats_low.float().unsqueeze(0))
        try:
            fused_float32 = self.attn(q, k, v_float32, need_weights=False)[0]
            fused = fused_float32.to(dtype=q.dtype)
        except Exception as e:
            print(f"!!! Error during MultiheadAttention: {e}")
            fused = q

        attn_mask = None
        if childof is not None:
            # childof: [N_H], each in [0, N_L-1]
            childof = childof.to(device=feats_high.device, dtype=torch.long).view(-1)
            N_H = childof.numel()
            N_L = feats_low.size(0)
    
            # MultiheadAttention attn_mask: shape [Lq, Lk], True means "disallow"
            attn_mask = torch.ones((N_H, N_L), dtype=torch.bool, device=feats_high.device)
            attn_mask[torch.arange(N_H, device=feats_high.device), childof] = False
    
        try:
            fused_float32 = self.attn(q, k, v_float32, attn_mask=attn_mask, need_weights=False)[0]
            fused = fused_float32.to(dtype=q.dtype)
        except Exception as e:
            print(f"!!! Error during MultiheadAttention: {e}")
            fused = q

        fused = fused + q
        fused_ffn = self.ffn(self.norm_out(fused))
        fused = fused + fused_ffn

        return fused.squeeze(0)

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = ChildGuidedCrossAttention(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feats_high, feats_low, childof_high):
        fused_high = self.cross_attn(feats_high, feats_low, childof_high)
        alpha = self.gate(torch.cat([feats_high, fused_high], dim=-1))
        with torch.no_grad():
            a = alpha.detach()
            self.debug_alpha_stats = (
                float(a.mean()),float(a.std()),float(a.min()),float(a.max())
            )
        out = alpha * fused_high + (1 - alpha) * feats_high
        return out

class HGIA(Baseline):
    def __init__(self, args,state_dict_weights):

        super(HGIA,self).__init__(args,state_dict_weights)
        self.gnn1 = HyperGNNModel(c_in=self.c_in, c_hidden=self.c_hidden, c_out=self.c_hidden,
                                  num_layers=args.num_layers, dp_rate=args.dropout_rate,k_nearest=4,
                                  build_hypergraph=True, apply_head=True)
        self.gnn2 = HyperGNNModel(c_in=self.c_in, c_hidden=self.c_hidden, c_out=self.c_hidden,
                                  num_layers=args.num_layers, dp_rate=args.dropout_rate,k_nearest=4,
                                  build_hypergraph=True, apply_head=True)
        self.GNNlast = HyperGNNModel(c_in=self.c_hidden, c_hidden=self.c_hidden, c_out=self.c_hidden,
                                     num_layers=args.num_layers, dp_rate=args.dropout_rate,k_nearest=4,
                                     build_hypergraph=False, apply_head=True)

        if self.residual:
            self.glu1 = GatedLinearUnit(self.c_in,self.c_in,norm=False)
            self.glu2 = GatedLinearUnit(self.c_in,self.c_in,norm=False)
            self.glulast = GatedLinearUnit(self.c_in,self.c_in,norm=False)
        else:
            self.glu1=None
            self.glu2=None
            self.glulast=None


        mil2fc,mil2bag = FCLayer(self.c_hidden,self.classes),BClassifier(self.c_hidden,self.classes)
        self.mil2 = MILNet(mil2fc,mil2bag)
        mil3fc,mil3bag = FCLayer(self.c_hidden,self.classes),BClassifier(self.c_hidden,self.classes)
        self.mil3 = MILNet(mil3fc,mil3bag)
        self.mil2 = init(self.mil2,self.state_dict_weights)
        self.mil3 = init(self.mil3,self.state_dict_weights)

        self.cross_fusion = GatedCrossAttentionFusion(dim=self.c_hidden)

    def forward_gnn(self,x,edge_index,levels,childof,edge_index2=None,edge_index3=None):

        results={}
        featsperlevel=[]
        indecesperlevel=[]
        # forward input for each scale gnn
        for i in levels.unique():
            #select scale
            indeces_feats=(levels==i).nonzero().view(-1)
            feats=x[indeces_feats]
            if i == levels.min():
                feats, _ = self.forward_scale(feats, edge_index2, self.gnn1, self.glu1)
                results["gnn_feats_lv0"] = feats.detach()
            else:
                feats, _ = self.forward_scale(feats, edge_index3, self.gnn2, self.glu2)
                results["gnn_feats_lv1"] = feats.detach()
            featsperlevel.append(feats)
            indecesperlevel.append(indeces_feats)
            torch.cuda.empty_cache()

        feats_low = featsperlevel[0]
        feats_high = featsperlevel[1]
        childof_for_attn = childof[indecesperlevel[1]]
        if feats_high.ndim == 3 and feats_high.size(0) == 1:
            feats_high = feats_high.squeeze(0)
        if feats_low.ndim == 3 and feats_low.size(0) == 1:
            feats_low = feats_low.squeeze(0)
        childof_high = childof_for_attn.view(-1)
        assert feats_high.size(0) == childof_high.size(0), \
            f"Mismatch: feats_high={feats_high.size(0)}, childof_high={childof_high.size(0)}"
        fused_high = self.cross_fusion(feats_high, feats_low, childof_for_attn)
        feats = torch.cat([feats_low, fused_high], dim=0)

        indeces= torch.concat(indecesperlevel).view(-1)
        indeces= indeces.sort()[1]

        if feats.ndim == 3:
            feats = feats[:, indeces, :]
        elif feats.ndim == 2:
            feats = feats[indeces]
        else:
            raise ValueError(f"Unsupported feats tensor shape: {feats.shape}")

        feats,edge_index2= self.forward_scale(feats, edge_index, self.GNNlast, self.glulast)

        results["childof"]=childof[indecesperlevel[1]]
        results["fused_high"] = fused_high

        return feats,indecesperlevel,results

    def forward_mil(self,indecesperlevel,feats,results):

        N = feats.shape[1]
        feats2d = feats.squeeze(0)

        childof = results["childof"].to(torch.long)

        valid_mask = (childof >= 0) & (childof < N)
        child_idx = torch.unique(childof[valid_mask])

        if child_idx.numel() > 0:
            feats_lower = feats2d[child_idx]
            results["lower"] = self.mil2(feats_lower.view(-1, self.c_hidden))
        else:
            print("results[\"lower\"] is none!")
            results["lower"] = self.mil2(feats2d.new_zeros((1, self.c_hidden)))

        high_idx = indecesperlevel[1].to(torch.long)
        high_idx = high_idx[(high_idx >= 0) & (high_idx < N)]
        if high_idx.numel() > 0:
            feats_higher = feats2d[high_idx]
            results["higher"] = self.mil3(feats_higher.view(-1, self.c_hidden))
        else:
            print("results[\"higher\"] is none!")
            results["higher"] = self.mil3(feats2d.new_zeros((1, self.c_hidden)))

        return results

