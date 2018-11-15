import random

import numpy as np
import requests

from py_grouping_estimates import groupingEstimates
from multiprocessing import Process


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SECONDS_TIMEOUT = 60 * 5


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


def fit_and_send_res():
    x_points, y_points = modulateRegression(100, OUTLIER_PERCENTAGE)
    APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)

    try:

        result = "GEM {}".format(APPROXIMATION_MODEL.fit())

        data_for_req = {
            "run_res": result
        }
        req = requests.post(url='https://py-cw-results.herokuapp.com/pushRunResult', json=data_for_req)

        print("%s: %s: %s:" % (result, req.status_code, req.reason))
    except Exception as e:
        print("Exception occurred: %s" % e)
    finally:
        pass


if __name__ == "__main__":
    while True:
        action_process = Process(target=fit_and_send_res)

        action_process.start()
        action_process.join(timeout=SECONDS_TIMEOUT)

        action_process.terminate()
