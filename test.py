import torch
import numpy as np
from metrics import computeMetrics
from sklearn.metrics import f1_score, roc_auc_score
import wandb

def _safe_f1(test_labels, test_predictions, n_classes):
    tl = np.asarray(test_labels)
    tp = np.asarray(test_predictions)

    if n_classes == 1:
        if tl.ndim == 2 and tl.shape[1] > 1:
            y_true = tl.argmax(axis=1)
        else:
            y_true = tl.astype(int).reshape(-1)

        if tp.ndim == 2:
            if tp.shape[1] >= 2:
                y_prob = tp[:, 1]
            else:
                y_prob = tp.squeeze()
        else:
            y_prob = tp.reshape(-1)

        y_pred = (y_prob >= 0.5).astype(int)

        if y_pred.shape[0] != y_true.shape[0]:
            print(f"[WARN] F1 skipped: y_true({y_true.shape}) vs y_pred({y_pred.shape})")
            return 0.0

        return f1_score(y_true, y_pred)

    else:
        y_true = tl.astype(int).reshape(-1)
        y_pred = tp.argmax(axis=1).astype(int)

        return f1_score(
            y_true,
            y_pred,
            average="weighted"
        )


def test(model, testloader):

    model.eval()
    results = []
    test_predictions0 = []
    test_predictions1 = []
    test_labels = []
    names = []
    # Iterate over the test data
    for _, data in enumerate(testloader):
        data = data.cuda()
        x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level

        if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
            edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
        else:
            edge_index2 = None
            edge_index3 = None

        results = model(x, edge_index, level, childof,
                        edge_index2, edge_index3)
        bag_label = data.y.float().squeeze().cpu().numpy()

        if model.classes == 1:
            bag_label = data.y.float().squeeze().cpu().numpy()
            if bag_label == 1:
                bag_label = torch.LongTensor([[0, 1]]).float().squeeze().cpu().numpy()
            else:
                bag_label = torch.LongTensor([[1, 0]]).float().squeeze().cpu().numpy()
            test_labels.extend([bag_label])
        else:
            bag_label = int(data.y.long().view(-1).cpu().item())
            test_labels.append(bag_label)

        preds = model.predict(results)

        p0 = preds[0].detach().cpu()
        if model.classes > 1:
            if p0.dim() == 1:
                p0 = p0.unsqueeze(0)
            p0 = p0.squeeze(0).numpy()
        else:
            p0 = p0.squeeze().numpy()

        test_predictions0.append(p0)

        if preds[1] is not None:
            p1 = preds[1].detach().cpu()
            if model.classes > 1:
                if p1.dim() == 1:
                    p1 = p1.unsqueeze(0)
                p1 = p1.squeeze(0).numpy()
            else:
                p1 = p1.squeeze().numpy()
            test_predictions1.append(p1)

    test_labels = np.array(test_labels)
    test_predictions0 = np.array(test_predictions0)
    test_predictions1 = np.array(test_predictions1)

    if model.classes == 1:

        avg_score_higher, auc_value_higher, class_prediction_bag_higher = computeMetrics(
            test_labels, test_predictions0, model.classes, names)

        if test_predictions1.shape[0] != 0:
            avg_score_lower, auc_value_lower, class_prediction_bag_lower = computeMetrics(
                test_labels, test_predictions1, model.classes, names)
        else:
            avg_score_lower, auc_value_lower, class_prediction_bag_lower = 0, 0, 0

        f1_higher = _safe_f1(test_labels, test_predictions0, model.classes)

    else:

        y_true = test_labels.astype(int).reshape(-1)
        logits = test_predictions0

        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

        y_pred = probs.argmax(axis=1)

        avg_score_higher = (y_pred == y_true).mean() * 100.0
        auc_value_higher = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        f1_higher = f1_score(y_true, y_pred, average="weighted")

        # 兼容返回值
        avg_score_lower, auc_value_lower, class_prediction_bag_lower = 0.0, 0.0, 0
        class_prediction_bag_higher = y_pred

    model.train()

    return avg_score_higher, avg_score_lower, auc_value_higher, auc_value_lower, class_prediction_bag_higher, class_prediction_bag_lower, test_labels, f1_higher, test_predictions0, test_predictions1
