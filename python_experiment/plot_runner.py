import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from py_grouping_estimates import groupingEstimates


ACCURATE_RESULT = np.matrix([90, 4, 7]).T
OUTLIER_PERCENTAGE = 8.0
SAMPLE_SIZE = 100
PLOT_SIZE = 30


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
    first_coordinates_with_classification = []
    second_coordinates_with_classification = []
    third_coordinates_with_classification = []
    first_coordinates_without_classification = []
    second_coordinates_without_classification = []
    third_coordinates_without_classification = []

    for iter_time in range(0, PLOT_SIZE):
        try:
            x_points, y_points = modulate_regression(SAMPLE_SIZE, OUTLIER_PERCENTAGE)

            APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)
            t_result_without = APPROXIMATION_MODEL.fit_without_reclassification()

            APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)
            t_result_with = APPROXIMATION_MODEL.fit()

            first_coordinates_without_classification.append(t_result_without[0])
            second_coordinates_without_classification.append(t_result_without[1])
            third_coordinates_without_classification.append(t_result_without[2])

            first_coordinates_with_classification.append(t_result_with[0])
            second_coordinates_with_classification.append(t_result_with[1])
            third_coordinates_with_classification.append(t_result_with[2])
        except np.linalg.linalg.LinAlgError as e:
            print(e)
        except StopIteration as e:
            print(e)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title("Оценки вектора [90, 4, 7]")
    ax.set_xlabel("beta_0")
    ax.set_ylabel("beta_1")
    ax.set_zlabel("beta_2")
    # ax.set_ax([80, 100, 3, 5, 6, 8])
    without_class = ax.scatter(first_coordinates_without_classification, second_coordinates_without_classification, third_coordinates_without_classification, color="green", marker="x")
    with_class = ax.scatter(first_coordinates_with_classification, second_coordinates_with_classification, third_coordinates_with_classification, color="blue", marker="s")
    accurate = ax.scatter(list(ACCURATE_RESULT[0]), list(ACCURATE_RESULT[1]), list(ACCURATE_RESULT[2]), color="red", marker="^")

    plt.legend((with_class, without_class, accurate),
               ('с переклассификацией', 'без переклассификации', 'истинное значение'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=6)
    plt.show()