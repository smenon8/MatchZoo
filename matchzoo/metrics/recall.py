""" Recall metric for ranking """
import math
import numpy as np
from matchzoo.engine.base_metric import BaseMetric, sort_and_couple


class Recall(BaseMetric):
    """ Class for calculating recall metric """

    ALIAS = ["recall"]

    def __init__(self, k: int = 1, threshold: float = 0.0):
        """
        :class:`DiscountedCumulativeGain` constructor.

        :param k: Number of results to consider default = 1.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        if self._k <= 0:
            raise ValueError(f"k must be greater than 0." f"{self._k} received.")

        coupled_pair = sort_and_couple(y_true, y_pred)

        pos_labels = 0
        results = 0
        for idx, (label, score) in enumerate(coupled_pair):
            if label > self._threshold:
                pos_labels += 1

                if idx < self._k:
                    results += 1

        if pos_labels == 0:
            return 0.0
        return pos_labels / results
