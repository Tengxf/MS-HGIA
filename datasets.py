import torch
import glob
import os
import json
import numpy as np
from torch_geometric.data import Dataset
import torch_geometric.loader as geom_loader
from seed import seed_worker
from sklearn.model_selection import StratifiedKFold

def _load_labels_json(datasetpath, processed_subdir="processed", json_name="labels.json"):
    proc_dir = os.path.join(datasetpath, processed_subdir)
    path = os.path.join(proc_dir, json_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found {path}")
    data = json.load(open(path, "r"))
    files  = data["files"]
    labels = np.array(data["labels"], dtype=np.int64)
    return proc_dir, files, labels

def get_loaders(args):
    proc_dir, all_files, y_all = _load_labels_json(args.datasetpath)
    if getattr(args, "fold", 0) == -1:
        print(f"🚀 [Inference Mode] Detected fold=-1. Using ALL {len(all_files)} files for testing.")
        train_files = []
        val_files = []
        test_files = all_files
    else:
        n_folds = 5

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        folds = list(skf.split(all_files, y_all))
            
        fold_id = getattr(args, "fold", 0)
        train_idx, test_idx = folds[fold_id]

        train_files = [all_files[i] for i in train_idx]
        test_files  = [all_files[i] for i in test_idx]
        val_files   = test_files

    def dump_fold_stats(name, files):
        all_labels = []
        for f in files:
            d = torch.load(os.path.join(proc_dir, f), map_location="cpu")
            y = int(d.y.view(-1)[0].item())
            all_labels.append(y)

        real_n_classes = max(args.n_classes, max(all_labels) + 1) if all_labels else args.n_classes
        counts = np.zeros(real_n_classes, dtype=int)

        for y in all_labels:
            counts[y] += 1

        print(f"[{name}] N={len(files)}  counts={counts.tolist()}")

        missing = np.where(counts == 0)[0].tolist()
        if len(missing) > 0:
            print(f"⚠️ Warning: [{name}] missing classes: {missing}")
            
    dump_fold_stats("TRAIN", train_files)
    dump_fold_stats("TEST",  test_files)
    dump_fold_stats("VAL",   val_files)

    # ---- DataLoader ----
    train_dataset = MyOwnDataset(root=args.datasetpath, files=train_files, type="train")
    val_dataset   = MyOwnDataset(root=args.datasetpath, files=val_files,   type="val")
    test_dataset  = MyOwnDataset(root=args.datasetpath, files=test_files,  type="test")

    if args.dataset == "ruxian":
        y_train = []
        for f in train_files:
            d = torch.load(os.path.join(proc_dir, f), map_location="cpu")
            y_train.append(int(d.y.item()))
        y_train = np.array(y_train)
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / np.sqrt(class_sample_count)
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    g_train = torch.Generator().manual_seed(args.seed + 101)
    g_val   = torch.Generator().manual_seed(args.seed + 102)
    g_test  = torch.Generator().manual_seed(args.seed + 103)

    if args.dataset == "ruxian":
        train_loader = geom_loader.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                              sampler=sampler, generator=g_train, worker_init_fn=seed_worker, pin_memory=True)
    else:
        train_loader = geom_loader.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                              generator=g_train, worker_init_fn=seed_worker, pin_memory=True)
    val_loader   = geom_loader.DataLoader(val_dataset,   batch_size=1, shuffle=True,
                                          generator=g_val,   worker_init_fn=seed_worker, pin_memory=True)
    test_loader  = geom_loader.DataLoader(test_dataset,  batch_size=1, shuffle=False,
                                          generator=g_test,  worker_init_fn=seed_worker, pin_memory=True)
    return train_loader, val_loader, test_loader

class MyOwnDataset(Dataset):
    def __init__(self, root=None, files=None, transform=None, pre_transform=None, type="train"):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.type = type
        if files is not None:
            root = os.path.join(root, "processed")
            self.bags = [os.path.join(root, f) for f in files] if not os.path.isabs(files[0]) else files
        else:
            self.bags = glob.glob(os.path.join(self.processed_dir, type, "*data*.pt"))

        self.data = [torch.load(bag) for bag in self.bags]
        self.length = len(self.bags)

    @property
    def processed_file_names(self):
        return glob.glob(os.path.join(self.processed_dir, "*"))

    def len(self):
        return self.length

    def get(self, idx):
        return self.data[idx]
