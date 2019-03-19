import numpy as np
import matplotlib.pyplot as plt

import big_data_runner

NP_DATA_PATH = "./np_data_created/"


def fit_data_naive_classic_visualise():
    sample_sizes = np.load(NP_DATA_PATH + "gem_sizes.npy")
    all_results_classic = np.load(NP_DATA_PATH + "gem_res_classic.npy")
    all_results_naive = np.load(NP_DATA_PATH + "gem_res_naive.npy")

    print(sample_sizes)
    print(all_results_classic)
    print(all_results_naive)

    plt.axis([0, sample_sizes[-1], 0, 10.0])

    plt.title("Сравнение аппроксимации с МНК по средним")
    plt.xlabel("размер выборки")
    plt.ylabel("вариации оценок")

    def generate_differences(_tarray):
        for item in _tarray:
            yield np.linalg.norm(big_data_runner.ACCURATE_RESULT.T - item.T)

    plot_classic, = plt.plot(sample_sizes,
                             list(generate_differences(all_results_classic)),
                             color="green")
    plt.axis([0, sample_sizes[-1], 0.0, 10.0])

    plot_naive, = plt.plot(sample_sizes,
                           list(generate_differences(all_results_naive)),
                           color="red")

    plt.legend((plot_classic, plot_naive),
               ('построенные оценки', 'МНК по средним'),
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()


def plot_with_different_sample_size_visualize():
    sample_sizes = np.load(NP_DATA_PATH + "gem_sizes_with_without.npy")
    all_results_with = np.load(NP_DATA_PATH + "gem_res_with.npy")
    all_results_without = np.load(NP_DATA_PATH + "gem_res_without.npy")

    print(sample_sizes)
    print(all_results_with)
    print(all_results_without)

    plt.title("Сравнение аппроксимаций c вкл/выкл классификацией")
    plt.xlabel("размер выборки")
    plt.ylabel("вариации оценок")

    def generate_differences(_tarray):
        for item in _tarray:
            yield np.linalg.norm(big_data_runner.ACCURATE_RESULT - item)

    plt.axis([0, sample_sizes[-1], 0, 10.0])
    plot_with, = plt.plot(sample_sizes,
                          list(generate_differences(all_results_with)),
                          color="green")
    plot_without, = plt.plot(sample_sizes,
                             list(generate_differences(all_results_without)),
                             color="red")

    plt.legend((plot_with, plot_without),
               ('с классификацией', 'без классификации'),
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()


def plot_with_different_reclassification_level():
    reclassification_levels = np.load(NP_DATA_PATH + "gem_with_dif_level_levels.npy")
    all_results_with_classification = np.load(NP_DATA_PATH + "gem_with_dif_level_results.npy")

    print(reclassification_levels)
    print(all_results_with_classification)

    plt.title("Сравнение разных уровней классификации")
    plt.xlabel("уровень классификации")
    plt.ylabel("вариации оценок")

    def generate_differences(_tarray):
        for item in _tarray:
            yield np.linalg.norm(big_data_runner.ACCURATE_RESULT - item)

    plt.axis([0, reclassification_levels[-1], 0, 10.0])
    plot_with, = plt.plot(reclassification_levels,
                          list(generate_differences(all_results_with_classification)),
                          color="green")

    plt.show()


if __name__ == "__main__":
    plot_with_different_sample_size_visualize()

    quit()
