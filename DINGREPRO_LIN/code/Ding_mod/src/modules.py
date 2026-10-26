# Resource:
# https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP
# https://github.com/mlfoundations/wise-ft
# https://github.com/LightDXY/FT-CLIP/tree/main
# https://github.com/damian0815/finetune-clip-huggingface/tree/main
import os
import numpy as np
import random
from PIL import Image
import math
import copy
import pandas as pd
import re

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import *
import torch.nn.functional as F

from sklearn.metrics import *
from fairlearn.metrics import *
import albumentations as A  # 导入albumentations库并将其缩写为A
from albumentations.pytorch import ToTensorV2  # 导入将图像转换为PyTorch张量的函数
import random
from torch.utils.data import DataLoader, Dataset

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if
             os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files
def find_all_files2(folder, suffix='npz'):
    if 'Training' in folder:
        folder = os.path.join(folder, 'train_npz')
    else:
        folder = os.path.join(folder, 'test_npz')
    files = [f for f in os.listdir(folder) if
             os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def truncate_note(note, max_length=180):
    # truncate the note if it is longer than 77 words, but maintain the word integrity
    if len(note) > max_length:
        note = note[:max_length]
        note = note[:note.rfind(' ')]

    return note


# 这个函数的目的是统计数据集中按种族、性别和民族划分的实例数量。它通过遍历数据集中的每个文件，
# 提取相关信息并统计每个类别中的实例数量。这对于分析数据集的分布和不平衡情况非常有用。
def count_number_of_groups(input_dataset):
    instances_on_race = []
    instances_on_gender = []
    instances_on_ethnicity = []
    for file in input_dataset.files:
        npz_path = os.path.join(input_dataset.dataset_dir, file)
        data = np.load(npz_path)
        instances_on_race.append(data['race'].item())
        instances_on_gender.append(data['gender'].item())
        instances_on_ethnicity.append(data['ethnicity'].item())
    # count the unique number in instances_on_race
    _, numbers_of_race = np.unique(instances_on_race, return_counts=True)
    _, numbers_of_gender = np.unique(instances_on_gender, return_counts=True)
    _, numbers_of_ethnicity = np.unique(instances_on_ethnicity, return_counts=True)
    return numbers_of_race, numbers_of_gender, numbers_of_ethnicity


class fair_vl_med_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='Training', text_source='note',
                 summarized_note_file=None, ruleout_unknown=False):
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.subset = subset
        self.text_source = text_source
        self.ruleout_unknown = ruleout_unknown

        self.summarized_notes = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary
        # 如果当前数据子集是训练集，并且文本来源是笔记，同时提供了总结笔记文件，
        if self.subset == 'Training' and self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))
            # df为gpt-4_summarized_notes.csv文件路径。包含1.npz和对应的笔记总结
            # 将每行的第一列值作为字典的键，将第三列值作为字典的值，存储在self.summarized_notes中
            for index, row in df.iterrows():
                self.summarized_notes[row.iloc[0].strip()] = row.iloc[2].strip()

        # check if the split file exists
        if files is not None:
            self.files = files
        else:  # 通过调用 find_all_files 函数在指定目录中查找所有 .npz 文件。
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            self.files = find_all_files(self.dataset_dir, suffix='npz')

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'Training' or self.ruleout_unknown:
            tmp_files = []
            for file in self.files:
                npz_path = os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and \
                        data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files

    def __len__(self):
        return len(self.files)

    # 如果self.text_source是'label'，根据青光眼标签生成描述并进行标记化。
    # 如果self.subset不是'Training'，生成青光眼和非青光眼的描述，并将它们的标记化结果连接成一个张量。
    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32)  # original size: (664, 512)  灰度图像
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))   #将 NumPy 数组 slo_fundus 转换为 PIL.Image 对象,进一步对图像操作

        if self.subset == 'Training':
            if self.text_source == 'note':

                note = self.summarized_notes[self.files[idx]].strip()
                # 截断处理
                note = truncate_note(note)
                token = clip.tokenize(note)
                token = token.squeeze()
            elif self.text_source == 'label':
                glaucoma_label = int(data['glaucoma'].item())
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                # 使用CLIP模型的tokenize方法将笔记转换为token（通常是一个适合模型输入的格式，如词嵌入或编码）
                # 然后使用squeeze方法去除不必要的维度。
                token = clip.tokenize(note)
                token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['ethnicity'].item())
        language = int(data['language'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language])

        return slo_fundus, token, label_and_attributes


class fair_vl_group_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='Training', text_source='note',
                 summarized_note_file=None, attribute='race', thegroup=0):
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.subset = subset
        self.text_source = text_source

        self.summarized_notes = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary
        if self.subset == 'Training' and self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))

            for index, row in df.iterrows():
                self.summarized_notes[row.iloc[0].strip()] = row.iloc[2].strip()

        # check if the split file exists
        if files is not None:
            self.files = files
        else:
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            self.files = find_all_files(self.dataset_dir, suffix='npz')

        # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
        # self.files = df[df['file_type'] == subset]['filename'].tolist()
        # self.files = files

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'Training':
            tmp_files = []
            for file in self.files:
                npz_path = os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and \
                        data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files
        #     groups_in_attrs = [3, 2, 2, 3]
        #     attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
        #     thegroup=0/1/2/3
        tmp_files = []
        for file in self.files:
            npz_path = os.path.join(self.dataset_dir, file)
            data = np.load(npz_path)

            group = int(data[attribute].item())
            if group == thegroup:
                tmp_files.append(file)
        self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32)  # original size: (664, 512)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'Training':
            if self.text_source == 'note':

                note = self.summarized_notes[self.files[idx]].strip()
                note = truncate_note(note)
                token = clip.tokenize(note)
                token = token.squeeze()
            elif self.text_source == 'label':
                glaucoma_label = int(data['glaucoma'].item())
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                token = clip.tokenize(note)
                token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['ethnicity'].item())
        language = int(data['language'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language])

        return slo_fundus, token, label_and_attributes


def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data


class image_title_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='train'):
        self.preprocess = preprocess
        self.files = files
        self.dataset_dir = dataset_dir
        self.subset = subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'train':
            note = truncate_note(data['note'].item().strip())
            token = clip.tokenize(note)
            token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['hispanic'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic])

        return slo_fundus, token, label_and_attributes


class Adversary_Net(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_sensitive),
        )

    def forward(self, x):
        return self.network(x)


# 计算视觉-语言的相似性概率和 logits得分
def compute_vl_prob(img_feats, class_txt_feats):
    # img_feats: [batch_size, 512]
    # class_txt_feats: [batch_size, num_class, 512]

    all_logits = []
    # 计算和第i个类别的相似性
    for i in range(class_txt_feats.shape[1]):
        similarity = (img_feats @ class_txt_feats[:, i, :].T)
        # extract the diagonal of the matrix，提取相似性矩阵的对角线元素，作为 logits
        logits = similarity.diag()

        all_logits.append(logits)
    # 堆叠所有的 logits，形成 [batch_size, num_class],代表每个图像与各个类别的相似性得分
    all_logits = torch.stack(all_logits, dim=1)

    # compute the probability by applying softmax along the second dimension
    # img_feats：[2, 512] class_txt_feats：[2, 2, 512]
    # vl_logits： tensor([[10.24, 9.76],
    #                   [9.56, 9.85]])
    # vl_logits 是相似性得分的对角线元素，表示每个图像和每个类别文本特征的相似性得分。
    vl_prob = torch.softmax(all_logits, dim=1)
    # [batch_size, num_class]
    return vl_prob, all_logits


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # 表示形状为一维，即是二元分类。统计正确样本数
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target) / target.shape[0]
        return acc.item()
    # 多元分类 output.topk(maxk, dim=1) 计算每个样本的 top-k 预测。
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        # 将 target 扩展为 [batch_size, 1],再变为[batch_size, maxk]
        target = target.view(batch_size, 1).repeat(1, maxk)

        correct = (pred == target)

        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size  # [0, 1.]
            topk_accuracy.append(accuracy)

        return topk_accuracy[0]


def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        # 比如y = torch.tensor([1, 1, 0, 2]),num_class=3->[[0 1 0] [0 1 0]  [1 0 0]  [0 0 1]]
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val


def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0] > 2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)

    return AUC


def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec


def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors


def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1

    return one_hot_array


def multiclass_demographic_parity(pred_prob, y, attrs):
    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:, i],
                                                  gt_one_hot[:, i],
                                                  sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score


def multiclass_equalized_odds(pred_prob, y, attrs):
    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:, i],
                                              gt_one_hot[:, i],
                                              sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score


def multiclass_demographic_parity_(pred_prob, y, attrs):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred == i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs == j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(
                tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]

    return mc_dpd


def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1] * 4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr - 0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr - 0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr - 0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr - 0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)

    return AUC, sensitivity_at_diff_specificity


# output为preds，公平性调整后的acc，先计算总体acc，然后计算每个组别的acc（比如遍历性别属性的男和女两个组别），
# 最后计算偏差和调整：计算每个组别的 AUC 值与总体 AUC 值之间的差异（绝对值），将这些差异的总和存储在 tmp 中。使用总体 AUC 除以 (1 + tmp) 进行调整，以得到公平性调整后的 AUC 值。
def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target) / target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target) / target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    # 比如性别：attrs为[1,1,1,0],np.unique(attrs).astype(int)为0,1
    # output = np.array([[0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])，对青光眼的预测
    # target = np.array([1, 0, 1, 0])
    # attrs = np.array([1, 1, 1, 0])
    # one_attr=0,attrs == one_attr的结果是 [True, False, False, True]，表示男（女）属性的预测的结果
    # pred_group = output[[True, False, False, True]]# 结果是 [[0.1, 0.9], [0.6, 0.4]]
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group) / gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group) / gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_acc)
    es_acc = (overall_acc / (alpha * tmp + 1))

    return es_acc


def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]
        if np.unique(gt_group).shape[0] == 1:
            print(f"Warning: Group {one_attr} has only one class. Skipping...")
            continue
        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_auc)
    es_auc = (overall_auc / (alpha * tmp + 1))

    return es_auc


def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    for i in range(attrs.shape[0]):
        attr = attrs[i, :]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append(compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes))
        aucs_by_attrs.append(aucs_by_group)
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                                pred_labels,
                                                sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                           pred_labels,
                                           sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                       pred_labels,
                                       sensitive_features=attr)

        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)


    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods


# 类别为2.默认为二元分类
# attrs：属性数组，形状为 [num_attributes, num_samples]。每个属性表示一个敏感特征（如种族、性别等）。
def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    ffprs = []
    foaes = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)
    # attrs.shape[0] （表示有几个属性,4），attrs[i,:]表示第i个属性的所有样本值
    attrs = attrs.reshape(1, -1)
    for i in range(attrs.shape[0]):
        attr = attrs[i, :]
        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)
        #筛选某一个属性的属性值，比如性别0或1  [0,1,0,1,0]->elements得到[0,1],preds值[0.1, 0.4, 0.35, 0.8, 0.6]，gts[0, 1, 0, 1, 1]
        #对于e = 0，筛选出attr == 0的样本，对应的preds是 [0.1, 0.35, 0.6]，对应的gts是 [0, 0, 1]。
        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            group_preds = preds[attr == e]
            group_gts = gts[attr == e]

            auc_val = compute_auc(group_preds, group_gts, num_classes=num_classes)

            if np.isnan(auc_val):
                continue
            aucs_by_group.append(auc_val)

        # aucs_by_attrs：用于存储每个属性的所有组别的 AUC 值列表，是 aucs_by_group 的集合，用于衡量多个属性上的性能。
        aucs_by_attrs.append(aucs_by_group)
        # 计算每个属性的组间标准差和最大差异。
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])
        # 计算公平性指标。pred_labels：根据预测概率生成的二分类标签
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                                pred_labels,
                                                sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                           pred_labels,
                                           sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                       pred_labels,
                                       sensitive_features=attr)
            ffpr = cal_ffpr_score(pred_labels, gts, attr)
            foae = cal_foae_score(pred_labels, gts, attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)
        ffprs.append(ffpr)
        foaes.append(foae)

        overall_acc = f"{overall_acc * 100:.2f}"
        overall_auc = f"{overall_auc * 100:.2f}"
        esaccs_by_attrs = [f"{es_acc * 100:.2f}" for es_acc in esaccs_by_attrs]
        esaucs_by_attrs = [f"{es_auc * 100:.2f}" for es_auc in esaucs_by_attrs]
        aucs_by_attrs = [[f"{auc * 100:.2f}" for auc in aucs] for aucs in aucs_by_attrs]
        dpds = [f"{dpd * 100:.2f}" for dpd in dpds]
        eods = [f"{eod * 100:.2f}" for eod in eods]
        ffprs = [f"{ffpr * 100:.2f}" for ffpr in ffprs]
        foaes = [f"{foae * 100:.2f}" for foae in foaes]
        between_group_disparity = [[f"{std * 100:.2f}", f"{max_d * 100:.2f}"] for std, max_d in
                                   between_group_disparity]
    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods,ffprs,foaes, between_group_disparity


def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i, :]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append(compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes))
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            # 用于计算不同组别之间的！！选择率差异，以评估模型在不同敏感特征组别上的公平性。选择率是指每个组别中被预测为正类的样本比例。
            # 先计算每个属性的选择率，再计算所有组别中选择率的最大值和最小值之间的差异。
            dpd = demographic_parity_difference(gts,
                                                pred_labels,
                                                sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                           pred_labels,
                                           sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                       pred_labels,
                                       sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity


# std_disparity：标准差差异度，即各组别 AUC 值的标准差与总体 AUC 值的比值
# max_disparity：最大差异度，即组别间最大和最小 AUC 值的差异与总体 AUC 值的比值。
def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list) - np.min(auc_list)) / overall_auc


def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc - 0.5), (np.max(auc_list) - np.min(auc_list)) / np.abs(
        overall_auc - 0.5)

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

def cal_foae_score(preds, labels, intersec_label):
    acc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            _, _, _, acc_section = classification_metrics(
                labels_section, pred_section)
            acc_all_sec.append(acc_section)
        except:
            continue
    return max(acc_all_sec)-min(acc_all_sec)

def cal_ffpr_score(preds, labels, intersec_label):
    efpr_s = 0.0

    logits_idx = (labels < 0.5)
    if np.sum(logits_idx) > 0:
        allg = np.sum(preds[logits_idx] == 1)/np.sum(logits_idx)
    else:
        allg = 0.0
        print("no real data in this batch")

    for j in list(np.unique(intersec_label)):
        group_neg_idx = (intersec_label == j) & (labels < 0.5)
        group_pos_idx = (intersec_label == j) & (labels >= 0.5)

        group_neg_count = np.sum(group_neg_idx)
        group_pos_count = np.sum(group_pos_idx)

        if group_neg_count == 0 or group_pos_count == 0:
            continue
        groupa = np.sum(preds[group_neg_idx] == 1)
        group = groupa / group_neg_count

        efpr_s += np.abs(group - allg)

    return efpr_s







































class fair_train(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, summarized_note_file=None, subset='Training'):
        self.preprocess = preprocess
        csv_file = os.path.join(dataset_dir, 'ff++', 'train.csv')
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id = row['id']
        img_path = str(row['img_path']).lstrip('/')
        base_path = "../data/Fairness_Generate"
        img_full_path = os.path.join(base_path, img_path)
        try:
            img = Image.open(img_full_path).convert('RGB')
            img = self.preprocess(img)
        except Exception as e:
            print(f"Error processing image {img_full_path}: {e}")

        deepfake_label = int(row['label'].item())
        intersec = int(row['intersec_label'].item())

        label_and_attributes = torch.tensor([deepfake_label, intersec])
        return img, label_and_attributes




class fair_group(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, subset='Training', summarized_note_file=None, attribute='intersec_label', thegroup=0):
        self.preprocess = preprocess
        csv_file = os.path.join(dataset_dir, 'ff++', 'train.csv')
        self.data = pd.read_csv(csv_file)

        tmp_files = []
        for index, row in self.data.iterrows():
            group = int(row[attribute])
            if group == thegroup:
                tmp_files.append(row)  # 将整个 row 添加到 tmp_files，或者只添加需要的字段
        self.files = tmp_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id = row['id']
        img_path = str(row['img_path']).lstrip('/')
        base_path = "../data/Fairness_Generate"
        img_full_path = os.path.join(base_path, img_path)
        try:
            img = Image.open(img_full_path).convert('RGB')
            img = self.preprocess(img)
        except Exception as e:
            print(f"Error processing image {img_full_path}: {e}")

        deepfake_label = int(row['label'].item())
        intersec = int(row['intersec_label'].item())

        label_and_attributes = torch.tensor([deepfake_label, intersec])
        return img, label_and_attributes


class fair_evaluate(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, subset='Test'):
        self.preprocess = preprocess

        csv_file = os.path.join(dataset_dir, 'dfdc', 'test.csv')
        # csv_file = os.path.join(dataset_dir, 'ffwithdfdc.csv')

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id = row['id']

        img_path = str(row['img_path']).lstrip('/')
        base_path = "../../data/Fairness_Generate"
        img_full_path = os.path.join(base_path, img_path)
        try:
            img = Image.open(img_full_path).convert('RGB')
            img = self.preprocess(img)
        except Exception as e:
            print(f"Error processing image {img_full_path}: {e}")

        deepfake_label = int(row['label'].item())
        intersec = int(row['intersec_label'].item())

        ismale = int(row['ismale'].item())
        isasian = int(row['isasian'].item())
        iswhite = int(row['iswhite'].item())
        isblack = int(row['isblack'].item())

        split = str(row['split'])


        isattrs= torch.tensor([ismale, isasian,iswhite,isblack])
        label_and_attributes = torch.tensor([deepfake_label, intersec])

        return id,img_path, img, label_and_attributes, isattrs, split