from typing import Tuple

import numpy as np


def calc_simple_metrics(gt: np.array, predictions: np.array, classes: iter) -> Tuple[float]:
    """
    :param gt: general truth
    :param predictions: predictions of the model, contains classes
    :param classes: list of size 2 that contains the classes of the objects (e.g. [0, 1])
    :return: TP, TN, FP, FN
    """
    TP = np.sum(gt[gt == classes[1]] == predictions[gt == classes[1]])
    TN = np.sum(gt[gt == classes[0]] == predictions[gt == classes[0]])
    FP = np.sum(gt[gt == classes[0]] != predictions[gt == classes[0]])
    FN = np.sum(gt[gt == classes[1]] != predictions[gt == classes[1]])

    return TP, TN, FP, FN

