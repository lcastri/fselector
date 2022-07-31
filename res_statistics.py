import datetime
import json
import os
from matplotlib import pyplot as plt
import numpy as np

_TIME = 'time'
_PREC = 'precision'
_RECA = 'recall'
_TEPCMCI = 'tepcmci'
_PCMCI = 'pcmci'
_F1SCORE = 'f1_score'

dlabel = {_TIME : 'time [s]',
          _PREC : 'precision',
          _RECA : 'recall',
           _F1SCORE : 'f1_score'}

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
        plt.xlabel("Iteration")
        plt.ylabel(dlabel[data])
        plt.legend(['TEPCMCI', 'PCMCI'])
        plt.title(data + ' comparison')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.eps')
        
        
def plot_statistics(resfolder):
    res_path = os.getcwd() + "/results/" + resfolder + "/res.json"
    plot_data(resfolder, res_path, _TIME)
    plot_data(resfolder, res_path, _F1SCORE)
    plot_data(resfolder, res_path, _PREC)
    plot_data(resfolder, res_path, _RECA)
    
    
    
    
    
def plot_data2(resfolder, file_path, data, nvar):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        data_tepcmci = list()
        data_pcmci = list()
        for i in r.keys():
            tmp_data_tepcmci = list()
            tmp_data_pcmci = list()
            for j in r[i].keys():
                if data == _TIME:
                    time_tepcmci = datetime.datetime.strptime(r[i][j][_TEPCMCI][data], '%H:%M:%S.%f')
                    time_pcmci = datetime.datetime.strptime(r[i][j][_PCMCI][data], '%H:%M:%S.%f')
                    tmp_data_tepcmci.append((time_tepcmci - since).total_seconds())
                    tmp_data_pcmci.append((time_pcmci - since).total_seconds())
                else:
                    tmp_data_tepcmci.append(r[i][j][_TEPCMCI][data])
                    tmp_data_pcmci.append(r[i][j][_PCMCI][data])
            if len(data_tepcmci) != 0:
                data_tepcmci = np.vstack([np.array(data_tepcmci), np.array(tmp_data_tepcmci)])
                data_pcmci = np.vstack([np.array(data_pcmci), np.array(tmp_data_pcmci)])
            else:
                data_tepcmci = np.array(tmp_data_tepcmci)
                data_pcmci = np.array(tmp_data_pcmci)
        
        data_tepcmci = np.sum(data_tepcmci, 0)
        data_tepcmci = np.divide(data_tepcmci, len(r))    
        data_pcmci = np.sum(data_pcmci, 0)
        data_pcmci = np.divide(data_pcmci, len(r))
            
        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(range(len(data_tepcmci)), data_tepcmci)
        plt.plot(range(len(data_pcmci)), data_pcmci)
        plt.xticks(range(len(data_tepcmci)))
        plt.xlabel("Iteration")
        plt.ylabel(dlabel[data])
        plt.legend(['TEPCMCI', 'PCMCI'])
        plt.title(data + ' comparison')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + str(nvar) + "_" + data + '.eps')
    
    
def plot_statistics2(resfolder, nvar):
    res_path = os.getcwd() + "/results/" + resfolder + "/" + str(nvar) + ".json"
    plot_data2(resfolder, res_path, _TIME, nvar)
    plot_data2(resfolder, res_path, _F1SCORE, nvar)
    plot_data2(resfolder, res_path, _PREC, nvar)
    plot_data2(resfolder, res_path, _RECA, nvar)
    
    
if __name__ == '__main__':   
    resfolder = 'TEvsPCMCI345'
    for nvar in range(3,6):
        plot_statistics2(resfolder, nvar)