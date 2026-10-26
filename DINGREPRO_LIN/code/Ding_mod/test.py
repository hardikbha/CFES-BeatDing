import os
import argparse
from detectors import DETECTOR
import torch
from torch.utils.data import DataLoader
from dataset.datasets_train import *
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import torch.nn.functional as F
from src import logger
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score
import os
import pandas as pd
import torch
from transform import fair_df_default_data_transforms as data_transforms
import torch.nn as nn

device = torch.device("cpu")
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--inter_attribute", type=str,default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others')
parser.add_argument("--result_dir", type=str, default='metricResults')
parser.add_argument("--save_dir", type=str, default='ff++Results/')

parser.add_argument("--checkpoints", type=str,
                    # default="Fairness-Generalization/training/checkpoints/daw_fdd_xception/ff++_/lamda1_0.1_lamda2_0.01_lr0.0005/")
                    # default="sink_results/resnet50_prune10/lr_1.0e-03_lambda_1.0e-01/")
                    default="/home/user/local/zhouyunpeng/SinkhornWithPrune/sink_results/XceptionDetector_prune10_ff++/lr_1.0e-03_lambda_9.0e-03")

parser.add_argument("--test_data_name", type=str, default='ff++')
parser.add_argument("--test_path", type=str, default="../data/ff++/test.csv")
parser.add_argument("--model_structure", type=str, default='xception_prune10_lambda_0.009', help="ori_resnet50|sinkhorn_resnet50|sinkhorn_xception|dag_resnet50|dag_xception|daw_resnet50")
# parser.add_argument("--model_structure", type=str, default='sinkhorn_prune_xception', help="ori_resnet50|sinkhorn_resnet50|sinkhorn_xception|dag_resnet50|dag_xception|daw_resnet50")
parser.add_argument("--epoch", type=int, default=200)
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False


# model_class = DETECTOR['daw_fdd']
# model_class = DETECTOR['dag_fdd']
# model_class = DETECTOR['Res50Detector']
model_class = DETECTOR['XceptionDetector']
model = model_class()
# model.backbone.last_linear = nn.Linear(2048, 2)
if cuda:
    model.cuda()


def sigmoid(inputarray):
    return 1/(1+np.exp(-1*inputarray))

def softmax(input_array, dim):
    # Shift input for numerical stability
    input_array = input_array - np.max(input_array, axis=dim, keepdims=True)

    # Calculate the exponential of the shifted array
    exp_values = np.exp(input_array)

    # Sum along the specified dimension
    sum_exp_values = np.sum(exp_values, axis=dim, keepdims=True)

    # Compute softmax values
    softmax_output = exp_values / sum_exp_values

    return softmax_output

def classification_metrics(label, prediction):
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

def acc_fairness(path, attribute_list):
    dict_data = {}

    dict_data['group_1'] = {}
    dict_data['group_1']['label'] = []
    dict_data['group_1']['logits'] = []
    dict_data['group_1']['prediction'] = []

    dict_data['group_2'] = {}
    dict_data['group_2']['label'] = []
    dict_data['group_2']['logits'] = []
    dict_data['group_2']['prediction'] = []

    length = len(attribute_list)
    if length == 2:
        for g1att in attribute_list[0]:#g1att:'nonmale'
            if g1att not in dict_data.keys():
                dict_data[g1att] = {}
                dict_data[g1att]['label'] = []
                dict_data[g1att]['logits'] = []
                dict_data[g1att]['prediction'] = []
            for g2att in attribute_list[1]:
                if g2att not in dict_data.keys():
                    dict_data[g2att] = {}
                    dict_data[g2att]['label'] = []
                    dict_data[g2att]['logits'] = []
                    dict_data[g2att]['prediction'] = []
                label_path = path+g1att+','+g2att+"labels.npy"
                # print(label_path, '')
                prediction_path = path+g1att+','+g2att+"predictions.npy"
                label = np.load(label_path)
                prediction = np.load(prediction_path)

                if len(label) != 0:
                    dict_data[g1att + ',' + g2att] = {}
                    dict_data[g1att + ',' + g2att]['label'] = label
                    dict_data[g1att + ',' + g2att]['logits'] = prediction
                    dict_data[g1att + ',' + g2att]['prediction'] = prediction
                    dict_data[g1att]['label'] += list(label)
                    dict_data[g1att]['logits'] += list(prediction)
                    dict_data[g1att]['prediction'] += list(prediction)
                    dict_data[g2att]['label'] += list(label)
                    dict_data[g2att]['logits'] += list(prediction)
                    dict_data[g2att]['prediction'] += list(prediction)
                    dict_data['group_1']['label'] += list(label)
                    dict_data['group_1']['logits'] += list(prediction)
                    dict_data['group_1']['prediction'] += list(prediction)
                    dict_data['group_2']['label'] += list(label)
                    dict_data['group_2']['logits'] += list(prediction)
                    dict_data['group_2']['prediction'] += list(prediction)

    metrics = {}
    for item in dict_data.keys():#male...
        # print(item, '1111111')
        metrics[item] = {}
        totalnumber = len(dict_data[item]['label'])
        if totalnumber != 0:
            try:
                auc, ErrorRate, FPR, TPR, acc, precision, F_G, eer, positive_rate, negative_rate, FNR, TNR = classification_metrics(
                    np.array(dict_data[item]['label']), np.array(dict_data[item]['prediction']))
                metrics[item]['auc'] = auc
                metrics[item]['ErrorRate'] = ErrorRate
                metrics[item]['FPR'] = FPR
                metrics[item]['TPR'] = TPR
                metrics[item]['acc'] = acc
                metrics[item]['precision'] = precision
                metrics[item]['F_G'] = F_G
                metrics[item]['eer'] = eer
                metrics[item]['positive_rate'] = positive_rate
                metrics[item]['negative_rate'] = negative_rate
                metrics[item]['FNR'] = FNR
                metrics[item]['TNR'] = TNR


                print(f' number of {totalnumber} with {item} : AUC: {auc:.5f} | ErrorRate: {ErrorRate: .5f} | fpr: {FPR:.5f} | tpr:{TPR:.5f} | acc: {acc:.5f} | precision: {precision:.5f} | EER: {eer:.5f} | F_G: {F_G:.5f}')
            except:
                print(
                    f' no positive or negative samples in the {totalnumber} of {item}, can not output the metrics')
        else:
            print(f' number of {item} is empty')
            continue
    overall_auc = metrics['group_1']['auc']
    return_results = []
    bbb = 0.0
    aaa = []
    efpr_inter = 0.0
    b_inter = 0.0
    a_inter = 0.0
    for names in metrics.keys():#names:'male,asian'
        # print(metrics.keys(), 'metrics.keys()')
        # print('---'+names+'---')
        if len(names.split(',')) >= 2 and metrics[names] != {}:
            if a_inter < np.abs(metrics[names]['F_G']-metrics['group_1']['F_G']):
                a_inter = np.abs(
                    metrics[names]['F_G']-metrics['group_1']['F_G'])

            b_inter += (np.abs(metrics[names]['FPR']-metrics['group_1']['FPR'])+np.abs(
                metrics[names]['TPR']-metrics['group_1']['TPR']))
            efpr_inter += (np.abs(metrics[names]
                           ['FPR']-metrics['group_1']['FPR']))

            aaa.append(metrics[names]['F_G'])

    for i in range(len(aaa)):
        for j in range(i+1, len(aaa)):
            if bbb < (np.abs(aaa[i]-aaa[j])):
                bbb = (np.abs(aaa[i]-aaa[j]))
    print(f' F_S: {bbb:.5f} ')
    F_A_inter = (bbb + a_inter) / 2
    print(
        f' F_G_inter, F_EFPR_inter, F_EO_inter, F_A_inter: {a_inter:.5f} | {efpr_inter:.5f} | {b_inter:.5f} | {F_A_inter:.5f} ')
    return_results += [metrics['group_1']['auc'], metrics['group_1']['FPR'],
                       metrics['group_1']['TPR'], metrics['group_1']['acc'], efpr_inter, b_inter]

    gap_g1, gap_g2, gap_inter, = {}, {}, {}
    gap_g1['auc'], gap_g1['ErrorRate'], gap_g1['FPR'], gap_g1['acc'], gap_g1['foae'], gap_g1['PR'], gap_g1['NR'], gap_g1['Fdp_diff'], gap_g1['F_DP'], gap_g1['FNR'], gap_g1['TNR'], gap_g1['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_g2['auc'], gap_g2['ErrorRate'], gap_g2['FPR'], gap_g2['acc'], gap_g2['foae'], gap_g2['PR'], gap_g2['NR'], gap_g2['Fdp_diff'], gap_g2['F_DP'], gap_g2['FNR'], gap_g2['TNR'], gap_g2['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_inter['auc'], gap_inter['ErrorRate'], gap_inter['FPR'], gap_inter['acc'], gap_inter['foae'], gap_inter['PR'], gap_inter['NR'], gap_inter['Fdp_diff'], gap_inter['F_DP'], gap_inter['FNR'], gap_inter['TNR'], gap_inter['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_g1['Fmeo_diff'], gap_g2['Fmeo_diff'], gap_inter['Fmeo_diff'] = [], [], []
    inter_auc_list = []
    g1_auc_list = []
    g2_auc_list = []
    g1_F_EFPR = 0.0
    g2_F_EFPR = 0.0
    for index, group in enumerate(attribute_list):
        a = 0.0
        b = 0.0
        efpr = 0.0
        cov = 0.0

        for weight, item in enumerate(group):
            if metrics[item] != {}:
                if a < np.abs(metrics[item]['F_G']-metrics['group_'+str(index+1)]['F_G']):
                    a = np.abs(metrics[item]['F_G'] -
                               metrics['group_'+str(index+1)]['F_G'])

                b += (np.abs(metrics[item]['FPR']-metrics['group_'+str(index+1)]['FPR'])+np.abs(
                    metrics[item]['TPR']-metrics['group_'+str(index+1)]['TPR']))
                efpr += np.abs(metrics[item]['FPR'] -
                               metrics['group_'+str(index+1)]['FPR'])
            if dict_data[item] != {}:
                firstitem = np.sum(
                    weight*dict_data[item]['logits'])/len(dict_data['group_'+str(index+1)]['label'])
                seconditem = np.multiply((weight*len(dict_data[item]['label'])/len(dict_data['group_'+str(
                    index+1)]['label'])), np.mean(dict_data['group_'+str(index+1)]['logits']))
                cov += np.abs(firstitem-seconditem)
                # print(cov)
        ccc = (a + bbb)/2
        print(
            f' F_G, F_EFPR, F_EO, F_A, F_COV of group_{index+1}: {a:.5f} | {efpr:.5f} | {b:.5f} | {ccc:.5f} | {cov:.5f}')
        if index == 0:
            g1_F_EFPR = efpr
        elif index == 1:
            g2_F_EFPR = efpr
        return_results += [efpr, b]

    for g1 in attribute_list[0]:
        try:
            gap_g1['auc'].append(metrics[g1]['auc'])
            g1_auc_list.append(metrics[g1]['auc'])
            gap_g1['ErrorRate'].append(metrics[g1]['ErrorRate'])
            gap_g1['FPR'].append(metrics[g1]['FPR'])
            gap_g1['acc'].append(metrics[g1]['acc'])
            gap_g1['foae'].append(metrics[g1]['acc'])
            gap_g1['PR'].append(metrics[g1]['positive_rate'])
            gap_g1['NR'].append(metrics[g1]['negative_rate'])
            gap_g1['FNR'].append(metrics[g1]['FNR'])
            gap_g1['TNR'].append(metrics[g1]['TNR'])
            gap_g1['TPR'].append(metrics[g1]['TPR'])

        except:
            continue
    tmp_g1 = sum(np.abs(np.array(g1_auc_list) - overall_auc))
    es_auc_g1 = overall_auc / (1.0 * tmp_g1 + 1)
    gap_g1['Fdp_diff'].append(max(gap_g1['PR'])-min(gap_g1['PR']))
    gap_g1['Fdp_diff'].append(max(gap_g1['NR'])-min(gap_g1['NR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['FPR'])-min(gap_g1['FPR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['FNR'])-min(gap_g1['FNR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['TNR'])-min(gap_g1['TNR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['TPR'])-min(gap_g1['TPR']))

    for g2 in attribute_list[1]:
        try:
            gap_g2['auc'].append(metrics[g2]['auc'])
            g2_auc_list.append(metrics[g2]['auc'])
            gap_g2['ErrorRate'].append(metrics[g2]['ErrorRate'])
            gap_g2['FPR'].append(metrics[g2]['FPR'])
            gap_g2['acc'].append(metrics[g2]['acc'])
            gap_g2['foae'].append(metrics[g2]['acc'])
            gap_g2['PR'].append(metrics[g2]['positive_rate'])
            gap_g2['NR'].append(metrics[g2]['negative_rate'])
            gap_g2['FNR'].append(metrics[g2]['FNR'])
            gap_g2['TNR'].append(metrics[g2]['TNR'])
            gap_g2['TPR'].append(metrics[g2]['TPR'])
        except:
            continue
    tmp_g2 = sum(np.abs(np.array(g2_auc_list) - overall_auc))
    es_auc_g2 = overall_auc / (1.0 * tmp_g2 + 1)
    gap_g2['Fdp_diff'].append(max(gap_g2['PR'])-min(gap_g2['PR']))
    gap_g2['Fdp_diff'].append(max(gap_g2['NR'])-min(gap_g2['NR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['FPR'])-min(gap_g2['FPR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['FNR'])-min(gap_g2['FNR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['TNR'])-min(gap_g2['TNR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['TPR'])-min(gap_g2['TPR']))

    for inter1 in attribute_list[0]:
        for inter2 in attribute_list[1]:
            try:
                gap_inter['auc'].append(metrics[inter1+','+inter2]['auc'])
                inter_auc_list.append(metrics[inter1 + ',' + inter2]['auc'])
                gap_inter['ErrorRate'].append(
                    metrics[inter1+','+inter2]['ErrorRate'])
                gap_inter['FPR'].append(metrics[inter1+','+inter2]['FPR'])
                gap_inter['acc'].append(metrics[inter1+','+inter2]['acc'])
                gap_inter['foae'].append(metrics[inter1+','+inter2]['acc'])
                gap_inter['PR'].append(
                    metrics[inter1+','+inter2]['positive_rate'])
                gap_inter['NR'].append(
                    metrics[inter1+','+inter2]['negative_rate'])
                gap_inter['FNR'].append(metrics[inter1+','+inter2]['FNR'])
                gap_inter['TNR'].append(metrics[inter1+','+inter2]['TNR'])
                gap_inter['TPR'].append(metrics[inter1+','+inter2]['TPR'])
            except:
                continue
    tmp_inter = sum(np.abs(np.array(inter_auc_list) - overall_auc))
    es_auc_inter = overall_auc / (1.0 * tmp_inter + 1)
    gap_inter['Fdp_diff'].append(max(gap_inter['PR'])-min(gap_inter['PR']))
    gap_inter['Fdp_diff'].append(max(gap_inter['NR'])-min(gap_inter['NR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['FPR'])-min(gap_inter['FPR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['FNR'])-min(gap_inter['FNR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['TNR'])-min(gap_inter['TNR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['TPR'])-min(gap_inter['TPR']))
    for i in ['FPR', 'auc', 'ErrorRate', 'foae']:
        try:
            print(f' inter_{i}, g1_{i}, g2_{i}: {(100*(max(gap_inter[i])-min(gap_inter[i]))):.3f} | {(100*(max(gap_g1[i])-min(gap_g1[i]))):.3f} | {(100*(max(gap_g2[i])-min(gap_g2[i]))):.3f} ')
            return_results += [(100*(max(gap_inter[i])-min(gap_inter[i]))), (100*(max(gap_g1[i])-min(gap_g1[i]))), (100*(max(gap_g2[i])-min(gap_g2[i])))]
        except:
            continue

    print(
        f' inter_F_DP,g1_F_DP,g2_F_DP:{100*max(gap_inter["Fdp_diff"]):.3f} | {100*max(gap_g1["Fdp_diff"]):.3f} | {100*max(gap_g2["Fdp_diff"]):.3f}')
    print(
        f' inter_F_meo,g1_F_meo,g2_F_meo:{100*max(gap_inter["Fmeo_diff"]):.3f} | {100*max(gap_g1["Fmeo_diff"]):.3f} | {100*max(gap_g2["Fmeo_diff"]):.3f}')


    inter_F_DP = 100 * max(gap_inter["Fdp_diff"])
    g1_F_DP = 100 * max(gap_g1["Fdp_diff"])
    g2_F_DP = 100 * max(gap_g2["Fdp_diff"])

    logger.logkv('auc',f"{overall_auc * 100:.2f}")
    logger.logkv('es_auc_inter', f"{es_auc_inter * 100:.2f}")
    logger.logkv('es_auc_g1', f"{es_auc_g1 * 100:.2f}")
    logger.logkv('es_auc_g2', f"{es_auc_g2 * 100:.2f}")
    logger.logkv('inter_F_DP', f"{inter_F_DP:.2f}")
    logger.logkv('g1_F_DP', f"{g1_F_DP:.2f}")
    logger.logkv('g2_F_DP', f"{g2_F_DP:.2f}")
    logger.logkv('inter_F_EFPR', f"{efpr_inter* 100:.2f}")
    logger.logkv('g1_F_EFPR', f"{g1_F_EFPR* 100:.2f}")
    logger.logkv('g2_F_EFPR', f"{g2_F_EFPR* 100:.2f}")

    return return_results

def main():
    save_path = os.path.join(opt.result_dir, opt.save_dir)
    log_dir = os.path.join(opt.result_dir, opt.test_data_name, opt.model_structure)
    logger.configure(dir=log_dir, log_suffix='fairness')
    checkpoint_files = sorted([f for f in os.listdir(opt.checkpoints) if f.endswith('.pth')])
    epochs = opt.epoch
    for i in range(epochs ):
        epo_num=i
        checkpoint_file = f"{i}.pth"
        if checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(opt.checkpoints, checkpoint_file)
            print(f'Loading checkpoint: {checkpoint_path}')
            # ckpt = torch.load(checkpoint_path, map_location=torch.device('cuda'), weights_only=False)
            # model.load_state_dict(ckpt)
            model = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

            nonmale_dst = []
            male_dst = []
            black_dst = []
            white_dst = []
            others_dst = []
            asian_dst = []

            interattributes = opt.inter_attribute.split('-')
            for eachatt in interattributes:
                # print(opt.test_path)
                test_dataset = ImageDataset_Test(
                    opt.test_path, eachatt, data_transforms['test'], opt.test_data_name)

                if 'nonmale,' in eachatt:
                    nonmale_dst.append(test_dataset)
                else:
                    male_dst.append(test_dataset)
                if ',black' in eachatt:
                    black_dst.append(test_dataset)
                if ',white' in eachatt:
                    white_dst.append(test_dataset)
                if ',others' in eachatt:
                    others_dst.append(test_dataset)
                if ',asian' in eachatt:
                    asian_dst.append(test_dataset)

                test_dataloader = DataLoader(
                    test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
                # print("%s" % opt.test_path)
                # print('Testing: ', eachatt)
                # print('-' * 10)
                # print('%d batches in total' % len(test_dataloader))

                pred_list = []
                label_list = []

                for i, data_dict in enumerate(test_dataloader):
                    bSTime = time.time()
                    model.eval()
                    data, label = data_dict['image'], data_dict["label"]
                    data_dict['image'], data_dict["label"] = data.to(device), label.to(device)

                    with torch.no_grad():
                        output = model(data_dict)
                        pred = F.softmax(output['cls'], dim=1)[:, 1]
                        pred = pred.cpu().data.numpy().tolist()

                        pred_list += pred
                        label_list += label.cpu().data.numpy().tolist()

                    bETime = time.time()
                    # print('#{} batch finished, eclipse time: {}'.format(i, bETime - bSTime))

                label_list = np.array(label_list)
                pred_list = np.array(pred_list)

                savepath = save_path + '/' + eachatt
                np.save(savepath + 'labels.npy', label_list)
                np.save(savepath + 'predictions.npy', pred_list)

            # acc_fairness('metricResults/', [['male', 'nonmale'], ['white', 'black', 'others']])
            acc_fairness(save_path, [['male', 'nonmale'], ['asian', 'white', 'black', 'others']])
            logger.logkv('epoch', f"{epo_num}")
            logger.dumpkvs()


if __name__ == '__main__':
    main()

## ff++ ####
# acc_fairness('results/', [
#              ['male', 'nonmale'], ['asian', 'white', 'black', 'others']])
# ### dfdc ###
# acc_fairness('results/',
#              [['male', 'nonmale'], ['asian', 'white', 'black', 'others']])
# # ### celebdf #####
# acc_fairness('results/',
#              [['male', 'nonmale'], ['white', 'black', 'others']])
# ### dfd ###
# acc_fairness('results/',
#              [['male', 'nonmale'], ['white', 'black', 'others']])

#test_data_name,test_path,acc_fairness,model_structure,checkpoints