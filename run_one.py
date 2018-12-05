import random

import numpy as np

from py_grouping_estimates import groupingEstimates

ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SECONDS_TIMEOUT = 60 * 5
SAMPLE_SIZE = 100
np.seterr(all='raise')


def alarm_handler(signum, frame):
    raise Exception("function timeout")


def modulateRegression(regression_sample_quintity, regression_outlier_percentage):
    regression_parameters = ACCURATE_RESULT
    _x_points = np.zeros(shape=[regression_sample_quintity, len(regression_parameters)])
    _y_points = np.zeros(shape=regression_sample_quintity)

    for i in range(0, regression_sample_quintity):
        if random.random() > regression_outlier_percentage / 100:
            _x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regression_parameters) - 1))
            _y_points[i] = (_x_points[i] * regression_parameters) + np.random.normal(0, 4)
        else:
            _x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regression_parameters) - 1))
            _y_points[i] = np.random.normal(100.0, 15.0, size=1)

    return _x_points, _y_points


def fit_and_print():
    x_points, y_points = modulateRegression(SAMPLE_SIZE, OUTLIER_PERCENTAGE)
    APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)
    result = "GEM {}".format(APPROXIMATION_MODEL.fit())
    print(result)


if __name__ == "__main__":
    fit_and_print()
    quit()
