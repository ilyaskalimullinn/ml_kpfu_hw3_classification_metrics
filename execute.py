import copy
import os

import pandas as pd

import visualisation
from dataset.sportsmans_height import Sportsmanheight
from metrics.binary_metrics import calc_binary_metrics
from model.simple_classifier import Classifier


def pr(metrics_list):
    metrics_list.sort(key=lambda m: m['recall'])

    auc = 0

    for i in range(len(metrics_list) - 1, 0, -1):
        metrics_list[i - 1]['precision'] = max(metrics_list[i - 1]['precision'], metrics_list[i]['precision'])
        # можно было использовать sklearn.metrics.auc, но я проверил, значения практически такие же.
        # чуть ниже мы добавляем две точки, но они не будут иметь большого влияния на площадь под кривой.
        auc += (metrics_list[i]['recall'] - metrics_list[i - 1]['recall']) * metrics_list[i]['precision']

    metrics_list.append({
        'recall': 1,
        'precision': 0
    })
    metrics_list.insert(0, {
        'recall': 0,
        'precision': 0
    })

    metrics = pd.DataFrame(metrics_list)

    visualisation.plot_precision_recall_curve(metrics,
                                              plot_title=f"Precision-Recall curve, AUC (Average precision) is {round(auc, 4)}",
                                              save_path=os.path.join(ROOT_DIR, 'graphs/PR.html'))


def roc(metrics_list):
    metrics_list.sort(key=lambda m: m['FPR'])

    auc = 0

    for i in range(1, len(metrics_list)):
        metrics_list[i]['recall'] = max(metrics_list[i - 1]['recall'], metrics_list[i]['recall'])
        auc += (metrics_list[i]['FPR'] - metrics_list[i - 1]['FPR']) * metrics_list[i - 1]['recall']

    metrics_list.insert(0, {
        'recall': 0,
        'FPR': 0
    })

    metrics = pd.DataFrame(metrics_list)

    visualisation.plot_roc_curve(metrics,
                                 plot_title=f"Receiver-Operating Characteristic curve, AUC (Average precision) is {round(auc, 4)}",
                                 save_path=os.path.join(ROOT_DIR, 'graphs/ROC.html'))


def experiment(gt, confidence):
    metrics_list = []

    for threshold in confidence:
        metrics_list.append(calc_binary_metrics(gt, confidence, threshold).__dict__)

    return metrics_list


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.curdir)

    dataset = Sportsmanheight()()
    confidence = Classifier()(dataset['height'])
    gt = dataset['class']

    metrics_list = experiment(gt, confidence)

    pr(copy.deepcopy(metrics_list))
    roc(metrics_list)
