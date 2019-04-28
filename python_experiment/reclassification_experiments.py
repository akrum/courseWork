import random
import os
import numpy as np
import itertools
from collections import namedtuple

from py_grouping_estimates import groupingEstimates
from py_grouping_estimates import groupingEstimatesNaive
from py_grouping_estimates.GroupingEstimatesDefines import GroupingEstimatesDefines

ACCURATE_RESULT = np.matrix([90, 4]).T
RegressionResult = namedtuple("RegressionResult", ["exogen", "endogen", "true_endogen", "outliers"])
OUTLIER_PERCENTAGE = 8.0
SAMPLE_SIZE = 100


def modulate_regression(regression_sample_quintity, regression_outlier_percentage):
    regression_parameters = ACCURATE_RESULT
    _x_points = np.zeros(shape=[regression_sample_quintity, len(regression_parameters)])
    _y_points = np.zeros(shape=regression_sample_quintity)
    _y_points_true = np.zeros(shape=regression_sample_quintity)
    outliercount = 0

    for i in range(0, regression_sample_quintity):
        _x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regression_parameters) - 1))
        _y_points_true[i] = (_x_points[i] * regression_parameters) + np.random.normal(0, 4)

        if random.random() > regression_outlier_percentage / 100:
            _y_points[i] = _y_points_true[i]
        else:
            _y_points[i] = np.random.normal(100.0, 15.0, size=1)

            outliercount += 1

    return RegressionResult(exogen=_x_points,
                            endogen=_y_points,
                            true_endogen=_y_points_true,
                            outliers=outliercount)


def compare_regressions():
    regression_model = modulate_regression(SAMPLE_SIZE, OUTLIER_PERCENTAGE)
    print("modulated with outlier count: %i" % regression_model.outliers)

    estimates_with_outliers = groupingEstimates.GEM(regression_model.exogen, regression_model.endogen)
    estimates_ordinary = groupingEstimates.GEM(regression_model.exogen, regression_model.true_endogen)

    estimates_with_outliers.classify()
    estimates_with_outliers.reclassify(GroupingEstimatesDefines.RECLASSIFICATION_LEVEL)

    estimates_ordinary.classify()

    def _list_comparator(list1, list2):
        for i in range(len(list1)):
            yield list1[i] == list2[i]

    difference_in_classified_positive = list(_list_comparator(estimates_ordinary._np_freq_positive, estimates_with_outliers._np_freq_positive))
    difference_in_classified_negative = list(_list_comparator(estimates_ordinary._np_freq_negative, estimates_with_outliers._np_freq_negative))
    full_difference_classified = difference_in_classified_positive.count(False) + difference_in_classified_negative.count(False)

    difference_in_reclassified_positive = list(_list_comparator(estimates_ordinary._np_freq_positive, estimates_with_outliers._np_freq_positive_reclassified))
    difference_in_reclassified_negative = list( _list_comparator(estimates_ordinary._np_freq_negative, estimates_with_outliers._np_freq_negative_reclassified))
    full_difference_reclassified = difference_in_reclassified_positive.count(False) + difference_in_reclassified_negative.count(False)

    print("Classes differ with/without outliers when without reclassification: %i" % full_difference_classified)
    print("Classes differ with/without outliers when with reclassification: %i" % full_difference_reclassified)


if __name__ == "__main__":
    compare_regressions()
    quit()
