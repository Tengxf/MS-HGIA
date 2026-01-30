import copy
import itertools
def launch_DASMIL_cam(args):
    args1= copy.copy(args)
    args1.modeltype="HGIA"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="23"
    args1.dataset="cam"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=256
    args1.layer_name="GAT"
    args1.residual=False
    return [args1]

def launch_DASMIL_lung(args):
    args1= copy.copy(args)
    args1.modeltype="HGIA"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="23"
    args1.dataset="lung"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=384
    args1.layer_name="GAT"
    args1.residual=True
    return [args1]

def launch_DASMIL_ruxian(args):
    args1= copy.copy(args)
    args1.modeltype="HGIA"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.0
    args1.scale="12"
    args1.dataset="ruxian"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.0001
    args1.c_hidden=256
    args1.layer_name="GAT"
    args1.residual=False
    args1.n_classes=4
    args1.dropout_rate=0
    return [args1]