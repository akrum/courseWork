import copy
import math

from .GroupingEstimatesDefines import GroupingEstimatesDefines as Defines
from threading import Thread
from multiprocessing import Event
from contextlib import contextmanager

import numpy as np
import sys


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class ApproximationGEMModelRedesigned():
    def __init__(self, exogen_data, endogen_data):
        self.exogen = exogen_data
        self.endogen = endogen_data
        self._np_freq_positive = None
        self._np_freq_negative = None
        self._np_freq_positive_reclassified = None
        self._np_freq_negative_reclassified = None
        self._shared_stop_event = Event()

    def erf(self, value):
        return math.sqrt(1.0 - math.exp(
            (-1.0 * value * value) * (4.0 / math.pi + Defines.a * value * value) / (1.0 + Defines.a * value * value)))

    def derf(self, value):
        if value == 0.0:
            return 2.0 / math.sqrt(math.pi)

        temp = math.exp((-1 * value * value) * (4 / math.pi + Defines.a * value * value) / (1 + Defines.a * value * value))
        temp *= ((-2 * value + 2 * Defines.a * value * value * value) * (4 / math.pi + Defines.a * value * value) / (
                1 + Defines.a * value * value) - 2 * Defines.a * value * value * value / (1 + Defines.a * value * value))
        temp /= 2 * math.sqrt(1 - math.exp(
            (-1 * value * value) * (4 / math.pi + Defines.a * value * value) / (1 + Defines.a * value * value)))

        return temp

    def _prob_func(self, x_i, y_i, mu_i, beta_hat, is_positive=True):
        a_mu_i_plus_1 = float('nan')
        a_mu_i = float('nan')

        if is_positive:
            if mu_i == Defines.K_EVERY_SEGMENT:
                return 0.5 * (1 + self.erf(
                    (Defines.intervals_right_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))))
            a_mu_i_plus_1 = mu_i * Defines.INTERVAL_LENGTH
            a_mu_i = mu_i * Defines.INTERVAL_LENGTH - Defines.INTERVAL_LENGTH
        else:
            if mu_i == Defines.K_EVERY_SEGMENT:
                return 0.5 * (1 + self.erf(
                    (Defines.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))))
            a_mu_i_plus_1 = -mu_i * Defines.INTERVAL_LENGTH
            a_mu_i = -mu_i * Defines.INTERVAL_LENGTH - Defines.INTERVAL_LENGTH

        return 0.5 * (self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) - self.erf(
            (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))))

    def _dprob_func(self, x_i, y_i, mu_i, beta_hat, is_positive=True):
        a_mu_i_plus_1 = float('nan')
        a_mu_i = float('nan')
        if is_positive:
            if mu_i == Defines.K_EVERY_SEGMENT:
                return 2.0 * x_i * self.derf(
                    (Defines.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) / (1.0 + self.erf(
                    (Defines.intervals_right_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ)))).item((0, 0))

            a_mu_i_plus_1 = mu_i * Defines.INTERVAL_LENGTH
            a_mu_i = mu_i * Defines.INTERVAL_LENGTH - Defines.INTERVAL_LENGTH

        else:
            if mu_i == Defines.K_EVERY_SEGMENT:
                return 2.0 * x_i * self.derf(
                    (Defines.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) / (1.0 + self.erf(
                    (Defines.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ)))).item((0, 0))

            a_mu_i_plus_1 = -mu_i * Defines.INTERVAL_LENGTH
            a_mu_i = -mu_i * Defines.INTERVAL_LENGTH - Defines.INTERVAL_LENGTH

        if self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) - self.erf(
                (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) == 0.0:
            return 0

        temp = 0.0
        temp += self.derf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) - self.derf(
            (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ)))
        temp /= self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ))) - self.erf(
            (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * Defines.SIGMA_SQ)))
        return 2.0 * x_i * temp.item((0, 0))

    def _likelihood_f(self, beta, mu_data, is_positive=True):
        current_likelihood_result = 0.0

        for i in range(0, self.endogen.size):
            if mu_data[i] is not None:
                temp_log_value = np.log(
                    self._prob_func(self.exogen[i], self.endogen[i], mu_data[i], beta, is_positive))
                if (np.isnan(temp_log_value) == False).all():
                    current_likelihood_result += temp_log_value

        return current_likelihood_result

    def _likelihood_f_without_log(self, beta, mu_data, is_positive=True):
        current_likelihood_result = 0.0

        for i in range(0, self.endogen.size):
            if mu_data[i] is not None:
                temp_value = self._prob_func(self.exogen[i], self.endogen[i], mu_data[i], beta, is_positive)
                if (np.isnan(temp_value) == False).all():
                    current_likelihood_result *= temp_value

        return current_likelihood_result

    def _dlikelihood_f(self, beta, mu_data, is_positive=True):
        current_likelihood_derivative_result = np.zeros(self.exogen[0].size)

        for i in range(0, self.endogen.size):
            if mu_data[i] is not None:
                current_likelihood_derivative_result += self._dprob_func(self.exogen[i], self.endogen[i], mu_data[i],
                                                                         beta, is_positive)

        return -0.5 * current_likelihood_derivative_result

    def full_cl_recl_likelihood_f(self, beta):
        return self._likelihood_f_without_log(beta, self._np_freq_positive_reclassified,
                                              is_positive=True) + self._likelihood_f_without_log(
            beta, self._np_freq_negative_reclassified, is_positive=False)

    def full_cl_recl_dlikelihood_f(self, beta):
        return self._dlikelihood_f(beta, self._np_freq_positive_reclassified, is_positive=True) + self._dlikelihood_f(
            beta, self._np_freq_negative_reclassified, is_positive=False)

    def full_cl_dlikelihood_f(self, beta):
        return self._dlikelihood_f(beta, self._np_freq_positive, is_positive=True) + self._dlikelihood_f(
            beta, self._np_freq_negative, is_positive=False)

    def classify(self):
        if self._np_freq_positive is not None:
            print("fit: classified yet positive")
            return
        if self._np_freq_negative is not None:
            print("fit: classified yet negative")
            return
        self._np_freq_positive = [None for i in range(self.endogen.size)]
        self._np_freq_negative = [None for i in range(self.endogen.size)]

        for i in range(self.endogen.size):
            if self.endogen[i] >= 0:
                self._np_freq_positive[i] = int(self.endogen[i] / Defines.INTERVAL_LENGTH)
            else:
                self._np_freq_negative[i] = int(abs(self.endogen[i]) / Defines.INTERVAL_LENGTH)

        print("fit: classified")

        return

    def reclassify(self, reclassify_K):
        if self._np_freq_positive_reclassified is not None:
            print("fit: reclassified yet")
            return
        if self._np_freq_negative_reclassified is not None:
            print("fit: reclassified yet")
            return

        self._np_freq_positive_reclassified = [None for i in range(self.endogen.size)]
        self._np_freq_negative_reclassified = [None for i in range(self.endogen.size)]

        count_reclassified = 0

        for i in range(0, self.endogen.size):
            distances = [None for _ in range(self.endogen.size)]

            for j in range(0, self.endogen.size):
                if self._np_freq_positive[j] is not None:
                    distances[j] = (np.linalg.norm(self.exogen[i]-self.exogen[j]), True, self._np_freq_positive[j])
                else:
                    distances[j] = (np.linalg.norm(self.exogen[i] - self.exogen[j]), False, self._np_freq_negative[j])

            sorted_distances = sorted(distances, key=lambda item: item[0])

            current_faced_classes_positive = {}
            current_faced_classes_negative = {}

            for j in range(0, reclassify_K):
                if sorted_distances[j][1] == True:
                    if sorted_distances[j][2] in current_faced_classes_positive:
                        current_faced_classes_positive[sorted_distances[j][2]] += 1
                    else:
                        current_faced_classes_positive[sorted_distances[j][2]] = 1
                if sorted_distances[j][1] == False:
                    if sorted_distances[j][2] in current_faced_classes_negative:
                        current_faced_classes_negative[sorted_distances[j][2]] += 1
                    else:
                        current_faced_classes_negative[sorted_distances[j][2]] = 1

            maximumfacedtimes_positive = 1 if self._np_freq_positive[i] is not None else 0
            maximumfacedclass_positive = self._np_freq_positive[i]
            maximumfacedtimes_negative = 1 if self._np_freq_negative[i] is not None else 0
            maximumfacedclass_negative = self._np_freq_negative[i]

            for key in current_faced_classes_positive:
                if maximumfacedtimes_positive < current_faced_classes_positive[key]:
                    maximumfacedtimes_positive = current_faced_classes_positive[key]
                    maximumfacedclass_positive = key

            for key in current_faced_classes_negative:
                if maximumfacedtimes_negative < current_faced_classes_negative[key]:
                    maximumfacedtimes_negative = current_faced_classes_negative[key]
                    maximumfacedclass_negative = key

            if maximumfacedtimes_positive > maximumfacedtimes_negative:
                self._np_freq_positive_reclassified[i] = maximumfacedclass_positive

                if self._np_freq_positive[i] is not None:
                    if maximumfacedclass_positive != self._np_freq_positive[i]:
                        count_reclassified += 1
                elif self._np_freq_negative[i] is not None:
                    if maximumfacedtimes_positive != self._np_freq_negative[i]:
                        count_reclassified += 1
            else:
                self._np_freq_negative_reclassified[i] = maximumfacedclass_negative

                if self._np_freq_positive[i] is not None:
                    if maximumfacedclass_negative != self._np_freq_positive[i]:
                        count_reclassified += 1
                elif self._np_freq_negative[i] is not None:
                    if maximumfacedclass_negative != self._np_freq_negative[i]:
                        count_reclassified += 1

        print("fit: reclassified: %i" % count_reclassified)

    def fit(self):
        print("")
        self.classify()
        self.reclassify(Defines.RECLASSIFICATION_LEVEL)

        print("fit: fitting.....")

        t_beta_hat = np.matrix([80.0, 0.0]).T
        t_beta_hat_next = np.matrix([100.0, 10.0]).T

        #############
        # FOR DEBUG #
        #############

        return self.fit_intercept(beta_hat=t_beta_hat, beta_hat_next=t_beta_hat_next)

        #############
        # FOR DEBUG #
        #############

        right_bound_indent = Defines.right_bound_fit_indent(self.exogen[0].size)
        loop_indentantion_value = Defines.LEFT_BOUND_EVERY_VAR_INDENT
        loop_end_bound = Defines.fit_loop_stop_value(self.exogen[0].size)

        beta_hats_left_bound = Defines.left_bound_fit_init(self.exogen[0].size)

        def recursive_beta_generator(index, previous_step_beta):
            assert (index <= self.exogen[0].size)

            beta_next = np.matrix.copy(previous_step_beta)

            while ((beta_next + right_bound_indent) < loop_end_bound).all():
                if index == self.exogen[0].size:
                    yield beta_next
                    break

                beta_next[index] += loop_indentantion_value

                next_index_generator = recursive_beta_generator(index + 1, beta_next)
                for item in next_index_generator:
                    yield item

        fit_intercept_results = []

        def fit_intercept_and_add_to_results(beta_hat_one, beta_hat_two):
            t_result = self.fit_intercept(beta_hat_one, beta_hat_two)
            if (np.isnan(t_result) == False).all():
                print("fit: added value to list %s" % t_result)
                fit_intercept_results.append(t_result)
            else:
                raise Exception("fit: got nan")

        created_threads = []
        for beta_left in recursive_beta_generator(0, beta_hats_left_bound):
            beta_right = beta_left + right_bound_indent
            calculus_thread = Thread(target=fit_intercept_and_add_to_results, args=(np.matrix.copy(beta_left),
                                                                                    np.matrix.copy(
                                                                                        beta_right)))
            created_threads.append(calculus_thread)
            calculus_thread.start()

        for thread in created_threads:
            thread.join(timeout=Defines.THREAD_JOIN_TIMEOUT)

        maximum_likelihood_res = None
        result_to_return = np.matrix([None for _ in range(self.exogen[0].size)]).T

        print("fit: possible betas: ")
        print(fit_intercept_results)

        initial_left_bound = Defines.left_bound_fit_init(self.exogen[0].size)
        initial_right_bound = loop_end_bound

        self._shared_stop_event.set()

        for result in fit_intercept_results:
            if (result < initial_left_bound).any():
                continue

            if (result > initial_right_bound).any():
                continue

            t_likelihood_res = self.full_cl_recl_likelihood_f(result)
            # print(t_likelihood_res)
            if maximum_likelihood_res is None:
                maximum_likelihood_res = t_likelihood_res
                result_to_return = result

            if maximum_likelihood_res < t_likelihood_res:
                maximum_likelihood_res = t_likelihood_res
                result_to_return = result

        return result_to_return

    def fit_intercept(self, beta_hat=None, beta_hat_next=None):
        if beta_hat is None:
            beta_hat = np.matrix(np.zeros(self.exogen[0].size)).T

        if beta_hat_next is None:
            beta_hat_next = np.matrix(np.ones(self.exogen[0].size)).T

        iteration_counter = 0
        while np.linalg.norm(beta_hat - beta_hat_next) > Defines.METHOD_ACCURACY:
            if self._shared_stop_event.is_set():
                raise StopIteration("fit: got stop event")
            if iteration_counter < Defines.COUNT_LIMIT_OPERATIONS:
                iteration_counter += 1
                dlikelihood_f_for_beta_hat_next = self.full_cl_recl_dlikelihood_f(beta_hat_next)
                delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T

                dlikelihood_derivative_approximation = np.zeros((self.exogen[0].size, self.exogen[0].size))

                for i in range(self.exogen[0].size):
                    temp_beta = copy.deepcopy(beta_hat_next)
                    temp_beta[i] = beta_hat[i]
                    # FIXME: something bad with dimensions
                    dlikelihood_derivative_approximation[i] = ((self.full_cl_recl_dlikelihood_f(
                        beta_hat_next) - self.full_cl_recl_dlikelihood_f(temp_beta)) / (beta_hat_next[i] - beta_hat[i])).A1

                delta_beta = (- np.matrix(dlikelihood_f_for_beta_hat_next)[0] * np.linalg.inv(
                    dlikelihood_derivative_approximation))  # FIXME: something wrong here
                beta_hat = beta_hat_next
                beta_hat_next = beta_hat_next + delta_beta.T
            else:
                raise StopIteration("fit: fit_intercept has achieved iteration limit")

        return beta_hat_next

    @contextmanager
    def turn_off_reclassification(self):
        self._np_freq_positive_reclassified = self._np_freq_positive
        self._np_freq_negative_reclassified = self._np_freq_negative
        yield
        print("fit: resetting reclassification")
        self._np_freq_positive_reclassified = None
        self._np_freq_negative_reclassified = None
        self._shared_stop_event.clear()

    def fit_without_reclassification(self):
        self.classify()
        with self.turn_off_reclassification():
            result_to_return = self.fit()
        return result_to_return


def GEM(exogen_data, endogen_data, *args, **kwargs):
    if "recl_level" in kwargs:
        print("gem: using recl level: {}".format(kwargs["recl_level"]))
        Defines.RECLASSIFICATION_LEVEL = kwargs["recl_level"]

    return ApproximationGEMModelRedesigned(exogen_data, endogen_data)
