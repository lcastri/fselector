import datetime
import json
import os
from matplotlib import pyplot as plt

_TIME = 'time'
_PREC = 'precision'
_RECA = 'recall'
_TEPCMCI = 'tepcmci'
_PCMCI = 'pcmci'
_F1SCORE = 'f1_score'

def get_TP(gt, cm):
    """
    True positive rate:
    edge present in the causal model 
    and present in the groundtruth

    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: true positive
    """
    counter = 0
    for node in cm.keys():
        for edge in cm[node]:
            if edge in gt[node]: counter += 1
    return counter


def get_FP(gt, cm):
    """
    False positive rate:
    edge present in the causal model 
    but absent in the groundtruth

    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: false positive
    """
    counter = 0
    for node in cm.keys():
        for edge in cm[node]:
            if edge not in gt[node]: counter += 1
    return counter


def get_FN(gt, cm):
    """
    False negative rate:
    edge present in the groundtruth 
    but absent in the causal model
    
    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: false negative
    """
    counter = 0
    for node in gt.keys():
        for edge in gt[node]:
            # if (node not in cm) or (edge not in cm[node]): counter += 1
            if edge not in cm[node]: counter += 1
    return counter


def precision(gt, cm):
    tp = get_TP(gt, cm)
    fp = get_FP(gt, cm)
    if tp + fp == 0: return 0
    return tp/(tp + fp)

    
def recall(gt, cm):
    tp = get_TP(gt, cm)
    fn = get_FN(gt, cm)
    if tp + fn == 0: return 0
    return tp/(tp + fn)


def f1_score(p, r):
    if p + r == 0: return 0
    return (2 * p * r) / (p + r)
        
        
def plot_data(resfolder, file_path, data):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        data_tepcmci = list()
        data_pcmci = list()
        for i in r.keys():
            if data == _TIME:
                time_tepcmci = datetime.datetime.strptime(r[i][_TEPCMCI][data], '%H:%M:%S.%f')
                time_pcmci = datetime.datetime.strptime(r[i][_PCMCI][data], '%H:%M:%S.%f')
                data_tepcmci.append((time_tepcmci - since).total_seconds())
                data_pcmci.append((time_pcmci - since).total_seconds())
            else:
                data_tepcmci.append(r[i][_TEPCMCI][data])
                data_pcmci.append(r[i][_PCMCI][data])
            
        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(range(len(r.keys())), data_tepcmci)
        plt.plot(range(len(r.keys())), data_pcmci)
        plt.legend(['TEPCMCI', 'PCMCI'])
        plt.title(data + ' comparison')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.eps')
        
        
def plot_statistics(resfolder):
    res_path = os.getcwd() + "/results/" + resfolder + "/res.json"
    plot_data(resfolder, res_path, _TIME)
    plot_data(resfolder, res_path, _F1SCORE)
    plot_data(resfolder, res_path, _PREC)
    plot_data(resfolder, res_path, _RECA)