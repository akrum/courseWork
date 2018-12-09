import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from py_grouping_estimates import groupingEstimates


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SAMPLE_SIZE_STEP = 10
PLOT_SIZE = 100
MEANING_FACTOR = 100


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
    differences = []
    sample_sizes = []
    for i in range(0, PLOT_SIZE):
        try:
            this_samle_size = 80 + i * SAMPLE_SIZE_STEP
            x_points, y_points = modulate_regression(this_samle_size, OUTLIER_PERCENTAGE)

            APPROXIMATION_MODEL = groupingEstimates.GEM(x_points, y_points)
            t_result = APPROXIMATION_MODEL.fit()

            differences.append(np.linalg.norm(t_result-ACCURATE_RESULT))
            sample_sizes.append(this_samle_size)
        except np.linalg.linalg.LinAlgError as e:
            print(e)
        except StopIteration as e:
            print(e)


plt.title("Зависимость точности от размера выборки")
plt.xlabel("размер выборки")
plt.ylabel("точность")
# plt.axis([80, 100, 3, 5])

sns.lineplot(sample_sizes, differences)
plt.show()
