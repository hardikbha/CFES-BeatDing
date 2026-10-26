import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score
from torch.nn.modules.batchnorm import _BatchNorm




def print_memory_usage(step):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{step}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
def auc_gap(preds, labels, intersec_label):
    auc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            auc_section, _, _, _ = classification_metrics(
                labels_section, pred_section)
            auc_all_sec.append(auc_section)
        except:
            continue
    return max(auc_all_sec)-min(auc_all_sec)
def cal_foae_score(preds, labels, intersec_label):
    acc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]#类别j的所有预测值
        labels_section = labels[intersec_label == j]#类别j的所有真实值
        try:
            _, _, _, acc_section = classification_metrics(
                labels_section, pred_section)
            acc_all_sec.append(acc_section)
        except:
            continue
    return max(acc_all_sec)-min(acc_all_sec)
def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    return auc, TPR, FPR, acc
def cal_ffpr_score(preds, labels, intersec_label):
    efpr_s = 0.0

    logits_idx = (labels < 0.5)
    if np.sum(logits_idx) > 0:
        allg = np.sum(preds[logits_idx] == 1)/np.sum(logits_idx)
    else:
        allg = 0.0
        print("no real data in this batch")

    for j in list(np.unique(intersec_label)):
        groupa = 0.0
        groupb = np.sum(labels[intersec_label == j] < 0.5)

        if groupb != 0:
            groupa = np.sum(preds[(intersec_label == j) & (logits_idx)] == 1)
            group = groupa/groupb
        else:
            group = 0.0

        efpr_s += np.abs(group - allg)

    return efpr_s

def cal_fmeo_score(preds, labels, intersec_label):

    fpr_diffs = []
    fnr_diffs = []
    tnr_diffs = []
    tpr_diffs = []

    for j in np.unique(intersec_label):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            _, _, FPR, TPR, _, _, _, _, _, _, FNR, TNR = clfi_metrics(labels_section, pred_section)
            fpr_diffs.append(FPR)
            fnr_diffs.append(FNR)
            tnr_diffs.append(TNR)
            tpr_diffs.append(TPR)
        except:
            continue

    if len(fpr_diffs) > 0 and len(fnr_diffs) > 0 and len(tnr_diffs) > 0 and len(tpr_diffs) > 0:
        fpr_diff = max(fpr_diffs) - min(fpr_diffs)
        fnr_diff = max(fnr_diffs) - min(fnr_diffs)
        tnr_diff = max(tnr_diffs) - min(tnr_diffs)
        tpr_diff = max(tpr_diffs) - min(tpr_diffs)
    else:
        fpr_diff = fnr_diff = tnr_diff = tpr_diff = 0.0

    # 取四个差值中的最大值并乘以100
    fmeo_score =  max(fpr_diff, fnr_diff, tnr_diff, tpr_diff)
    return fmeo_score

def cal_fdp_score(preds, labels, intersec_label):
    """
    计算Fdp评分，衡量不同交叉群体之间的PR和NR的差异，并乘以100返回。
    """
    pr_diffs = []
    nr_diffs = []

    for j in np.unique(intersec_label):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            _, _, _, _, _, _, _, _, positive_rate, negative_rate, _, _ = clfi_metrics(labels_section,
                                                                                                pred_section)
            pr_diffs.append(positive_rate)
            nr_diffs.append(negative_rate)
        except:
            continue

    if len(pr_diffs) > 0 and len(nr_diffs) > 0:
        pr_diff = max(pr_diffs) - min(pr_diffs)
        nr_diff = max(nr_diffs) - min(nr_diffs)
    else:
        pr_diff = nr_diff = 0.0

    # 取PR和NR差值中的最大值并乘以100
    fdp_score = max(pr_diff, nr_diff)
    return fdp_score
def clfi_metrics(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fnr = 1-tpr
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr-fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]

    auc = roc_auc_score(label, prediction)

    acc = accuracy_score(label, prediction >= 0.5)

    precision = precision_score(label, prediction >= 0.5)

    CM = confusion_matrix(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)

    FNR = FN/(FN+TP)
    TNR = TN/(TN+FP)
    f_g = (TP+FP)/(len(label))

    positive_rate = (TP+FP)/(len(label))
    negative_rate = (TN+FN)/(len(label))

    er = 1-((TPR+(1-FPR))/2)

    return auc, er, FPR, TPR, acc, precision, f_g, eer, positive_rate, negative_rate, FNR, TNR

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

    def enable_running_stats(model):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)