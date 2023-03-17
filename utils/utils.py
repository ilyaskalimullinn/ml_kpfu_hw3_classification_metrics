from typing import Tuple

import numpy as np


def sort_confidence_and_gt(confidence: np.array, gt: np.array, reverse=False) -> Tuple[np.array, np.array]:
    """
    :param confidence:
    :param gt:
    :param reverse:
    :return: confidence and general truth sorted
    """
    p = confidence.argsort()
    if reverse:
        p = np.flip(p)
    return confidence[p], gt[p]
