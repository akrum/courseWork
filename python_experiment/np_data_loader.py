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

    plt.title("Сравнение аппроксимации с простейшей")
    plt.xlabel("размер выборки")
    plt.ylabel("вариации оценок")

    def generate_differences(_tarray):
        for item in _tarray:
            yield np.linalg.norm(big_data_runner.ACCURATE_RESULT - item)

    # plt.axis([0, 10.0, 0, sample_sizes[-1]])
    plot_classic, = plt.plot(sample_sizes,
                            list(generate_differences(all_results_classic)),
                            color="green")
    plot_naive, = plt.plot(sample_sizes,
                          list(generate_differences(all_results_naive)),
                          color="red")

    plt.legend((plot_classic, plot_naive),
               ('построенные оценки', 'МНК по средним'),
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()


if __name__ == "__main__":
    fit_data_naive_classic_visualise()

    quit()
