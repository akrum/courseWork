import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from py_grouping_estimates import groupingEstimates


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SAMPLE_SIZE = 100
PLOT_SIZE = 150


def alarm_handler(signum, frame):
    raise Exception("function timeout")


def modulate_regression(regression_sample_quintity, regression_outlier_percentage):
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


if __name__ == "__main__":
    first_coordinates = []
    second_coordinates = []
    for iter_time in range(0, PLOT_SIZE):
        x_points, y_points = modulate_regression(SAMPLE_SIZE, OUTLIER_PERCENTAGE)
        APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)
        t_result = APPROXIMATION_MODEL.fit()

        first_coordinates.append(t_result[0])
        second_coordinates.append(t_result[1])

    plt.title("Оценки вектора 90, 4")
    plt.xlabel("beta_0")
    plt.ylabel("beta_1")
    plt.axis([80, 100, 3, 5])
    sns.scatterplot(list(ACCURATE_RESULT[0]), list(ACCURATE_RESULT[1]), color="red")
    sns.scatterplot(first_coordinates, second_coordinates)
    plt.show()
