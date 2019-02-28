import random
import os

import numpy as np

from py_grouping_estimates import groupingEstimates
from py_grouping_estimates import groupingEstimatesNaive

ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SECONDS_TIMEOUT = 60 * 5
SAMPLE_SIZE_MIN = 100
SAMPLE_SIZE_MAX = 2000
SAMPLE_SIZE_STEP = 100
np.seterr(all='raise')
NP_DATA_PATH = "./np_data_created/"


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


def fit_data_naive_classic():
    sample_sizes = []
    all_results_classic = []
    all_results_naive = []
    for sample_size in range(SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX+1, SAMPLE_SIZE_STEP):
        x_points, y_points = modulateRegression(sample_size, OUTLIER_PERCENTAGE)
        approx_model = groupingEstimates.GEM(x_points, y_points)
        approx_model_naive = groupingEstimatesNaive.GEM_N(x_points, y_points)
        try:
            result = approx_model.fit()
            print("GEM {}".format(result))
            result_naive = approx_model_naive.fit()
            print("GEM_N {}".format(result_naive))
            all_results_classic.append(result)
            all_results_naive.append(result_naive)
            sample_sizes.append(sample_size)
        except KeyboardInterrupt:
            print("stopping...")
            quit()
        except Exception as e:
            print(e)
    np.save(NP_DATA_PATH + "gem_res_classic", all_results_classic)
    np.save(NP_DATA_PATH + "gem_res_naive", all_results_naive)
    np.save(NP_DATA_PATH + "gem_sizes", sample_sizes)


if __name__ == "__main__":
    if not os.path.exists(NP_DATA_PATH):
        os.makedirs(NP_DATA_PATH)
    fit_data_naive_classic()
    quit()
