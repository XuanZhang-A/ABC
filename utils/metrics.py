import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from torcheval.metrics import BinaryAUROC
from torcheval.metrics.functional import binary_precision_recall_curve, binary_confusion_matrix
import matplotlib.pyplot as plt


class OODMetrics:
    def __init__(self) -> None:
        self.y_preds = []
        self.y_trues = []
        self.auroc = BinaryAUROC()

    def update(self, y_pred, y_true):
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
    
    def reset(self):
        self.auroc.reset()
        self.y_preds = []
        self.y_trues = []
    
    def plot_histogram(self):
        y_scores = torch.cat(self.y_preds, dim=0)
        y_true = torch.cat(self.y_trues, dim=0)
        
        scores_class_0 = y_scores[y_true == 0]
        scores_class_1 = y_scores[y_true == 1]
        
        plt.hist(scores_class_0.cpu().numpy(), bins=50, alpha=0.7, label='ID', color='green')
        plt.hist(scores_class_1.cpu().numpy(), bins=50, alpha=0.7, label='OOD', color='red')
        
        plt.xlabel('Scores')
        plt.ylabel('Frequency')
        plt.title('Histogram of Predicted Scores')
        plt.legend()
        return plt

    def compute(self):
        # Calculate confusion matrix
        y_scores = torch.cat(self.y_preds, dim=0)
        y_true = torch.cat(self.y_trues, dim=0)

        # AUROC
        self.auroc.update(y_scores, y_true)
        auroc = self.auroc.compute()

        #  Compute precision, recall points and thresholds using torcheval
        precision_points, recall_points, thresholds = binary_precision_recall_curve(y_scores, y_true)
        
        # Calculate F1 score for each possible threshold
        f1_scores = 2 * (precision_points * recall_points) / (precision_points + recall_points)
        f1_scores = torch.nan_to_num(f1_scores)  # Convert NaNs to zero where precision + recall is zero

        # Find the threshold that gives the highest F1 score
        max_f1_index = torch.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_index]
        optimal_threshold = thresholds[max_f1_index]

        # Calculate metrics at optimal threshold
        y_pred = (y_scores >= optimal_threshold).int()
        y_pred = y_pred.to(torch.int64)
        y_true = y_true.to(torch.int64)

        # Calculate FPR
        tn, fp, fn, tp = binary_confusion_matrix(y_pred, y_true).ravel()
        fpr = fp.float() / (fp + tn) if (fp + tn) > 0 else torch.tensor(0.)

        # Calculate AUPR
        precision_points, recall_points, _ = binary_precision_recall_curve(y_scores, y_true)
        sorted_indices = torch.argsort(recall_points)
        sorted_recall_points = recall_points[sorted_indices]
        sorted_precision_points = precision_points[sorted_indices]
        aupr = torch.trapz(sorted_precision_points, sorted_recall_points)

        return {
            "AUROC": auroc,
            "Optimal_F1": max_f1,
            "FPR": fpr,
            "AUPR": aupr
        }


class OptimalF1:
    def __init__(self):
        super().__init__()
        self.preds = []
        self.targets = []
    
    def update(self, y_pred, y_true):
        self.preds.append(y_pred)
        self.targets.append(y_true)
    
    def compute(self):
        y_pred = torch.cat(self.preds, dim=0)
        y_true = torch.cat(self.targets, dim=0)
        precision, recall, thresholds = binary_precision_recall_curve(y_pred, y_true)
        img_f1s = 2 * precision * recall / (precision + recall)
        threshold = thresholds[torch.argmax(img_f1s)]
        self.reset()
        return img_f1s.max(), threshold

    def reset(self):
        self.preds = []
        self.targets = []


def accuracy(output, target, return_length=False, **kwargs):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)

def top_k_acc(output, target, k=5, return_length=False, **kwargs):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)

# ====from COCL====
# TODO 它使用的是numpy，真的没问题吗！
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for culmulative sum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))

def f1_score(y_score, y_true, num_classes=10, **kwargs):
    y_predict = torch.argmax(F.softmax(y_score, dim=1), dim=1)
    y_target = y_true

    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device=y_target.device)
    f1_score = f1(y_predict, y_target)
    return f1_score.item()

def false_positive_rate(y_score, y_true, num_classes=10, **kwargs):
    # y_true = y_true.detach().cpu().numpy()

    preds = torch.argmax(F.softmax(y_score, dim=1), dim=1)
    target = y_true

    fprs = []

    for class_label in range(num_classes):
        # False Positives (FP): predicted as class_label, but not actually class_label
        FP = torch.sum((preds == class_label) & (target != class_label))
        # True Negatives (TN): not class_label and not predicted as class_label
        TN = torch.sum((preds != class_label) & (target != class_label))
    
        # False Positive Rate (FPR)
        if FP + TN == 0:
            breakpoint()
            fprs.append(0)
        else:
            fpr = FP.float() / (FP + TN)
            fprs.append(fpr.item())
    return np.sum(fprs)


def get_measures(_pos, _neg, recall_level=0.95):
    """Calculate all measures
    =============
    Parameters
    ----
    _pos : 
        ID predicts
    _neg :
        OOD predicts
    recal_level : 
        The specified recall level for calculating the false positve rate (fpr)
    =============
    Return
    ----
    auroc : 
        Area Under the Receiver Operating Characteristic curve
    aupr_in : 
        Area Under the Precision-Recall curve for the positive class
    aupr_out : 
        Area Under the Precision-Recall curve for the negative class, using the negative of examples to treat negative instances as positive for calculation
    fpr : 
        False Positive Rate, at a specified recall level, and the mean scores for positive and negative examples.
    """
    if isinstance(_pos, torch.Tensor):
        _pos = _pos.detach().cpu().numpy()
        _neg = _neg.detach().cpu().numpy()
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr_in = average_precision_score(labels, examples)
    labels_rev = np.zeros(len(examples), dtype=np.int32)
    labels_rev[len(pos):] += 1
    aupr_out = average_precision_score(labels_rev, -examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr_in, aupr_out, fpr, pos.mean(), neg.mean()


'''
Adapted from https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/utils.py
which uses GPL-3.0 license.
'''
def shot_acc(preds, labels, train_class_count, acc_per_cls=False):

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    num_classes = len(train_class_count)

    test_class_count = [np.nan] * num_classes
    class_correct = [np.nan] * num_classes
    for l in range(num_classes):
        test_class_count[l] = len(labels[labels == l])
        class_correct[l] = (preds[labels == l] == labels[labels == l]).sum()

    # print(train_class_count, len(train_class_count))
    # print(test_class_count, len(test_class_count))
    # print(np.unique(labels))
    # print(test_class_count)
    # CIFAR10 rho=100: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    # CIFAR100 rho=100: [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 
    # 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 
    # 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 
    # 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 
    # 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]

    if num_classes <= 100: # e.g. On CIFAR10/100
        many_shot_thr = train_class_count[int(0.34*num_classes)]
        low_shot_thr = train_class_count[int(0.67*num_classes)]
    else:
        many_shot_thr=100
        low_shot_thr=20
    # print(many_shot_thr, low_shot_thr)

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(num_classes):
        if test_class_count[i] == 0:
            assert class_correct[i] == 0
            _acc_class_i = np.nan
        else:
            _acc_class_i = class_correct[i] / test_class_count[i]
        if train_class_count[i] > many_shot_thr:
            many_shot.append(_acc_class_i)
        elif train_class_count[i] < low_shot_thr:
            low_shot.append(_acc_class_i)
        else:
            median_shot.append(_acc_class_i)    

    # print('many_shot:', many_shot)
    # print('median_shot:', median_shot)
    # print('low_shot:', low_shot)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot), class_accs
    else:
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot)