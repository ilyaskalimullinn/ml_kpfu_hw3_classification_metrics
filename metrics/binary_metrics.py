from typing import Tuple

import numpy as np


class FullBinaryMetrics:
    threshold: float
    TP: float
    TN: float
    FP: float
    FN: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def __init__(self, TP, TN, FP, FN, threshold):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

        self.threshold = threshold

        self.accuracy = (TP + TN) / (TP + TN + FP + FN)
        if TP == 0:
            self.precision = 0
            self.recall = 0
            self.f1_score = 0
        else:
            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


def calc_binary_metrics(gt: np.array, confidence: np.array, threshold: float = 0.5,
                        classes: iter = [0, 1]) -> FullBinaryMetrics:
    """
    :param gt: general truth, actual classes
    :param confidence: confidence of the model that the object belongs to class 1
    :param classes: list of size 2 that contains the classes of the objects (default [0, 1])
    :param threshold: threshold for defining the class (values >= threshold are considered class 1)
    :return: full metrics
    """

    predictions = np.where(confidence > threshold, 1, 0)

    TP = np.sum(gt[gt == classes[1]] == predictions[gt == classes[1]])
    TN = np.sum(gt[gt == classes[0]] == predictions[gt == classes[0]])
    FP = np.sum(gt[gt == classes[0]] != predictions[gt == classes[0]])
    FN = np.sum(gt[gt == classes[1]] != predictions[gt == classes[1]])

    return FullBinaryMetrics(TP, TN, FP, FN, threshold)
