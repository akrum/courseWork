import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.api as sm


from py_grouping_estimates import groupingEstimates


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0
SAMPLE_SIZE = 100
SAMPLE_SIZE_MIN = 100
SAMPLE_SIZE_MAX = 1000
SAMPLE_SIZE_STEP = 10
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


def with_without_reclassification():
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
    without_class = ax.scatter(first_coordinates_without_classification, second_coordinates_without_classification,
                               third_coordinates_without_classification, color="green", marker="x")
    with_class = ax.scatter(first_coordinates_with_classification, second_coordinates_with_classification,
                            third_coordinates_with_classification, color="blue", marker="s")
    accurate = ax.scatter(list(ACCURATE_RESULT[0]), list(ACCURATE_RESULT[1]), list(ACCURATE_RESULT[2]), color="red",
                          marker="^")

    plt.legend((with_class, without_class, accurate),
               ('с переклассификацией', 'без переклассификации', 'истинное значение'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=6)
    plt.show()


def m_estimators():
    sample_sizes = []
    all_results_m_estim = []

    x_points = None
    y_points = None

    def _visualize():
        plt.title("Оценки параметров линейной регрессии c искажениями (М-оценки)")
        plt.xlabel("размер выборки")
        plt.ylabel("вариации оценок")

        def generate_differences(_tarray):
            for item in _tarray:
                yield np.linalg.norm(ACCURATE_RESULT.A1 - item)

        plt.axis([0, sample_sizes[-1], 0, 10.0])

        plot_without, = plt.plot(sample_sizes,
                                 list(generate_differences(all_results_m_estim)),
                                 color="red")
        plt.show()

    for sample_size in range(SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX + 1, SAMPLE_SIZE_STEP):
        print("fitting with sample size: {}".format(sample_size))
        successful_fit = False
        while not successful_fit:
            x_points_t, y_points_t = modulate_regression(SAMPLE_SIZE_STEP, OUTLIER_PERCENTAGE)

            if x_points is None or y_points is None:
                x_points, y_points = modulate_regression(SAMPLE_SIZE_MIN, OUTLIER_PERCENTAGE)
            else:
                x_points = np.append(x_points, x_points_t, axis=0)
                y_points = np.append(y_points, y_points_t, axis=0)

            approx_model = sm.RLM(y_points, x_points, M=sm.robust.norms.HuberT())
            try:
                result_without = approx_model.fit().params
                print("RLM {}".format(result_without))
                successful_fit = True

                all_results_m_estim.append(result_without)
                sample_sizes.append(sample_size)
            except KeyboardInterrupt:
                print("stopping...")
                _visualize()
                quit()
            except Exception as e:
                print(e)

    _visualize()


def ols_estimators():
    sample_sizes = []
    all_differences_m_estim = []

    x_points = None
    y_points = None

    for sample_size in range(SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX + 1, SAMPLE_SIZE_STEP):
        print("fitting with sample size: {}".format(sample_size))
        x_points_t, y_points_t = modulate_regression(SAMPLE_SIZE_STEP, OUTLIER_PERCENTAGE)

        full_dif = 0.0

        for i in range(0, 100):
            if x_points is None or y_points is None:
                x_points, y_points = modulate_regression(SAMPLE_SIZE_MIN, OUTLIER_PERCENTAGE)
            else:
                x_points_t = np.append(x_points, x_points_t, axis=0)
                y_points_t = np.append(y_points, y_points_t, axis=0)

            approx_model = sm.OLS(y_points_t, x_points_t, M=sm.robust.norms.HuberT())

            result_without = approx_model.fit().params
            print("RLM {}".format(result_without))
            full_dif += np.linalg.norm(ACCURATE_RESULT.A1 - result_without)

        full_dif /= 100
        all_differences_m_estim.append(full_dif)
        sample_sizes.append(sample_size)

    plt.title("Оценки параметров линейной регрессии c искажениями (МНК-оценки)")
    plt.xlabel("размер выборки")
    plt.ylabel("вариации оценок")

    plt.axis([0, sample_sizes[-1], 0, 10.0])

    plot_without, = plt.plot(sample_sizes,
                             all_differences_m_estim,
                             color="red")
    plt.show()


if __name__ == "__main__":
    ols_estimators()