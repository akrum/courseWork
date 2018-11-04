import numpy as np
from random import random
import math
import threading
import copy
import statsmodels.api as sm


ACCURATE_RESULT = np.matrix([90, 4]).T
OUTLIER_PERCENTAGE = 8.0


class ApproximationGEMModel:
    def __init__(self, exogen_data, endogen_data):
        self._a = 8.0 / (3.0 * math.pi) * (3.0 - math.pi) / (math.pi - 4.0)
        self.interval_length = 0.001
        self._k_every_segment = 50000000
        self.intervals_left_bound = 0.0 - self.interval_length * self._k_every_segment / 2.0
        self.intervals_right_bound = 0.0 + self.interval_length * self._k_every_segment / 2.0
        self.exogen = exogen_data
        self.endogen = endogen_data
        self.sigmasq = 225.0
        self.threadingEvent = threading.Event()
        self.mu_data = None
        self.mu_data_reclassified = None

    def erf(self, value):
        return math.sqrt(1.0 - math.exp((-1.0*value*value) * (4.0 / math.pi + self._a * value * value) / (1.0 + self._a * value * value)))

    def derf(self, value):

        temp = math.exp((-1*value*value) * (4 / math.pi + self._a * value * value) / (1 + self._a * value * value))
        temp *= ((-2 * value + 2 * self._a * value * value * value) * (4 / math.pi + self._a * value * value) / (1 + self._a * value * value) - 2 * self._a * value * value * value / (1 + self._a * value * value))
        temp /= 2 * math.sqrt(1 - math.exp((-1*value*value) * (4 / math.pi + self._a * value * value) / (1 + self._a * value * value)))

        return temp

    def _prob_func(self, x_i, y_i, mu_i, beta_hat):
        if mu_i == 0:
            return 0.5 * (1 + self.erf((self.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))))

        if mu_i == self._k_every_segment - 1:
            return 0.5 * (1 + self.erf((self.intervals_right_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))))

        a_mu_i_plus_1 = (mu_i - self._k_every_segment / 2) * self.interval_length
        a_mu_i = (mu_i - self._k_every_segment / 2) * self.interval_length - self.interval_length

        return 0.5 * (self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))-self.erf((a_mu_i - x_i * beta_hat)/(math.sqrt(2.0 * self.sigmasq))))

    def _dprob_func(self, x_i, y_i, mu_i, beta_hat):
        if mu_i == 0:
            return 2.0*x_i*self.derf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))).item((0, 0))

        if mu_i == self._k_every_segment-1:
            return 2.0*x_i*self.derf((self.intervals_right_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))).item((0, 0))

        a_mu_i_plus_1 = (mu_i - self._k_every_segment / 2) * self.interval_length
        a_mu_i = (mu_i - self._k_every_segment / 2) * self.interval_length - self.interval_length

        if self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.erf((a_mu_i - x_i*beta_hat) / (math.sqrt(2.0 * self.sigmasq))) == 0.0:
            return 0

        temp = 0.0
        temp += self.derf((a_mu_i_plus_1 - x_i * beta_hat)/(math.sqrt(2.0 * self.sigmasq))) - self.derf((a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))
        temp /= self.erf((a_mu_i_plus_1 - x_i*beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.erf((a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))

        return 2.0 * x_i * temp.item((0, 0))
    
    def _likelihood_f(self, beta, mu_data):
        current_likelihood_result = 0.0
        for i in range(0, self.endogen.size):
            current_likelihood_result += np.log(self._prob_func(self.exogen[i], self.endogen[i], mu_data[i], beta))
        return current_likelihood_result

    def _dlikelihood_f(self, beta, mu_data):
        current_likelihood_derivative_result = np.zeros(self.exogen[0].size)

        for i in range(0, self.endogen.size):
            current_likelihood_derivative_result+= self._dprob_func(self.exogen[i], self.endogen[i], mu_data[i], beta)

        return -0.5*current_likelihood_derivative_result

    def classify(self):
        # TODO: Неправильная классификация
        self.mu_data = np.zeros(self.endogen.size)
        for i in range(self.endogen.size):
            if self.endogen[i]<self.intervals_left_bound:
                self.mu_data[i]=0
            elif self.endogen[i]>self.intervals_right_bound:
                self.mu_data[i]= self._k_every_segment - 1
            else:
                self.mu_data[i] = int(round(self.endogen[i]/self.interval_length)) + self._k_every_segment / 2

        print("classified")

        return self

    def reclassify(self, delta):
        self.mu_data_reclassified = np.zeros(self.endogen.size)
        for i in range(0, self.endogen.size):
            current_faced_classes={}
            for j in range(0, self.endogen.size):
                if np.linalg.norm(self.exogen[i]-self.exogen[j])<=delta:
                    if self.mu_data[j] in current_faced_classes:
                        current_faced_classes[self.mu_data[j]]+=1
                    else:
                        current_faced_classes[self.mu_data[j]]=1

            maximumfacedtimes = 1
            maximumfacedclass = self.mu_data[i]

            for key in current_faced_classes:
                if maximumfacedtimes < current_faced_classes[key]:
                    maximumfacedtimes = current_faced_classes[key]
                    maximumfacedclass = key

            self.mu_data_reclassified[i] = maximumfacedclass

        print("reclassified")

        return self

    def fit(self):
        self.classify()
        self.reclassify(0.5)

        print("fitting.....")

        beta_hat = np.matrix(np.ones(self.exogen[0].size)).T
        beta_hat_next = np.matrix(np.zeros(self.exogen[0].size)).T

        while np.linalg.norm(self._dlikelihood_f(beta_hat_next, self.mu_data_reclassified)) >= 0.1:
            dlikelihood_f_for_beta_hat = self._dlikelihood_f(beta_hat, self.mu_data_reclassified)
            dlikelihood_f_for_beta_hat_next = self._dlikelihood_f(beta_hat_next, self.mu_data_reclassified)
            delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T

            for i in range(self.exogen[0].size):
                delta_beta[i] = -dlikelihood_f_for_beta_hat_next[i] / (dlikelihood_f_for_beta_hat_next[i] - dlikelihood_f_for_beta_hat[i]) * (beta_hat_next[i]-beta_hat[i])
            beta_hat = beta_hat_next
            beta_hat_next = beta_hat_next+delta_beta

        return beta_hat_next

    def fit_without_classification(self):
        self.classify()

        print("fitting.....")

        beta_hat = np.matrix(np.ones(self.exogen[0].size)).T
        beta_hat_next = np.matrix(np.zeros(self.exogen[0].size)).T

        while np.linalg.norm(self._dlikelihood_f(beta_hat_next, self.mu_data)) >= 0.1:
            dlikelihood_f_for_beta_hat = self._dlikelihood_f(beta_hat, self.mu_data)
            dlikelihood_f_for_beta_hat_next = self._dlikelihood_f(beta_hat_next, self.mu_data)
            delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T
            for i in range(self.exogen[0].size):
                delta_beta[i] = -dlikelihood_f_for_beta_hat_next[i] / (dlikelihood_f_for_beta_hat_next[i] - dlikelihood_f_for_beta_hat[i]) * (beta_hat_next[i]-beta_hat[i])
            beta_hat = beta_hat_next
            beta_hat_next = beta_hat_next+delta_beta

        return beta_hat_next

    def compare(self):
        self.classify()

        beta_hat = np.matrix([170, 8]).T
        print(self._dlikelihood_f(beta_hat, self.mu_data))

        without_classification = self._dlikelihood_f(ACCURATE_RESULT, self.mu_data)
        print(without_classification)

        temp_delta = 1.0
        self.reclassify(temp_delta)

        with_classification = self._dlikelihood_f(ACCURATE_RESULT, self.mu_data_reclassified)

        while np.equal(with_classification, ACCURATE_RESULT).all():
            temp_delta += 0.5
            with_classification = self._dlikelihood_f(ACCURATE_RESULT, self.mu_data_reclassified)
    
        print(self._dlikelihood_f(ACCURATE_RESULT, self.mu_data_reclassified))

        return without_classification, with_classification


class ApproximationGEMModelRedesigned(ApproximationGEMModel):
    def __init__(self, exogen_data, endogen_data):
        super(ApproximationGEMModelRedesigned, self).__init__(exogen_data, endogen_data)
        self._segment_count_in_every_array = self._k_every_segment
        self._np_freq_positive = None
        self._np_freq_negative = None
        self._np_freq_positive_reclassified = None
        self._np_freq_negative_reclassified = None
        self.METHOD_ACCURACY = 1e-5

    def _prob_func(self, x_i, y_i, mu_i, beta_hat, is_positive=True):
        a_mu_i_plus_1 = float('nan')
        a_mu_i = float('nan')

        if is_positive:
            if mu_i == self._k_every_segment:
                return 0.5 * (1 + self.erf((self.intervals_right_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))))
            a_mu_i_plus_1 = mu_i * self.interval_length
            a_mu_i = mu_i * self.interval_length - self.interval_length
        else:
            if mu_i == self._k_every_segment:
                return 0.5 * (1 + self.erf((self.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))))
            a_mu_i_plus_1 = -mu_i * self.interval_length
            a_mu_i = -mu_i * self.interval_length - self.interval_length

        return 0.5 * (self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.erf((a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))))

    def _dprob_func(self, x_i, y_i, mu_i, beta_hat, is_positive=True):
        a_mu_i_plus_1 = float('nan')
        a_mu_i = float('nan')
        if is_positive:
            if mu_i == self._k_every_segment:
                return 2.0 * x_i * self.derf((self.intervals_right_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) / (1.0 + self.erf((self.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))).item((0, 0))

            a_mu_i_plus_1 = mu_i * self.interval_length
            a_mu_i = mu_i * self.interval_length - self.interval_length

        else:
            if mu_i == self._k_every_segment:
                return 2.0 * x_i * self.derf((self.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) / (1.0 + self.erf((self.intervals_left_bound - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))).item((0, 0))

            a_mu_i_plus_1 = -mu_i * self.interval_length
            a_mu_i = -mu_i * self.interval_length - self.interval_length

        if self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.erf(
                (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) == 0.0:
            return 0

        temp = 0.0
        temp += self.derf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.derf(
            (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))
        temp /= self.erf((a_mu_i_plus_1 - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq))) - self.erf(
            (a_mu_i - x_i * beta_hat) / (math.sqrt(2.0 * self.sigmasq)))
        return 2.0 * x_i * temp.item((0, 0))

    def _likelihood_f(self, beta, mu_data, is_positive=True):
        current_likelihood_result = 0.0

        for i in range(0, self.endogen.size):
            if mu_data[i] is not None:
                current_likelihood_result += np.log(self._prob_func(self.exogen[i], self.endogen[i], mu_data[i], beta, is_positive))

        return current_likelihood_result

    def _dlikelihood_f(self, beta, mu_data, is_positive=True):
        current_likelihood_derivative_result = np.zeros(self.exogen[0].size)

        for i in range(0, self.endogen.size):
            if mu_data[i] is not None:
                current_likelihood_derivative_result += self._dprob_func(self.exogen[i], self.endogen[i], mu_data[i], beta, is_positive)

        return -0.5 * current_likelihood_derivative_result

    def full_cl_recl_dlikelihood_f(self, beta):
        return self._dlikelihood_f(beta, self._np_freq_positive_reclassified, is_positive=True) + self._dlikelihood_f(beta, self._np_freq_negative_reclassified, is_positive=False)

    def classify(self):
        self._np_freq_positive = [None for i in range(self.endogen.size)]
        self._np_freq_negative = [None for i in range(self.endogen.size)]

        for i in range(self.endogen.size):
            if self.endogen[i] >= 0:
                self._np_freq_positive[i] = int(self.endogen[i] / self.interval_length)
            else:
                self._np_freq_negative[i] = int(abs(self.endogen[i]) / self.interval_length)

        print("classified")

        return self

    def reclassify(self, delta):
        self._np_freq_positive_reclassified = [None for i in range(self.endogen.size)]
        self._np_freq_negative_reclassified = [None for i in range(self.endogen.size)]

        for i in range(0, self.endogen.size):
            current_faced_classes_positive = {}
            current_faced_classes_negative = {}
            for j in range(0, self.endogen.size):
                if np.linalg.norm(self.exogen[i]-self.exogen[j]) <= delta:
                    if self._np_freq_positive[j] is not None:
                        if self._np_freq_positive[j] in current_faced_classes_positive:
                            current_faced_classes_positive[self._np_freq_positive[j]] += 1
                        else:
                            current_faced_classes_positive[self._np_freq_positive[j]] = 1
                    elif self._np_freq_negative[j] is not None:
                        if self._np_freq_negative[j] in current_faced_classes_positive:
                            current_faced_classes_negative[self._np_freq_negative[j]] += 1
                        else:
                            current_faced_classes_negative[self._np_freq_negative[j]] = 1
                    else:
                        continue
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

            if maximumfacedclass_positive > maximumfacedtimes_negative:
                self._np_freq_positive_reclassified[i] = maximumfacedclass_positive
            else:
                self._np_freq_negative_reclassified[i] = maximumfacedclass_negative

        print("reclassified")

    def fit(self):
        self.classify()
        self.reclassify(0.5)

        print("fitting.....")

        beta_hat = np.matrix(np.ones(self.exogen[0].size)).T
        beta_hat_next = np.matrix(np.zeros(self.exogen[0].size)).T

        # FIXME: something bad with numerical method
        while np.linalg.norm(self._dlikelihood_f(beta_hat_next, self._np_freq_positive_reclassified, is_positive=True) + self._dlikelihood_f(beta_hat_next, self._np_freq_negative_reclassified, is_positive=False)) >= self.METHOD_ACCURACY:
            dlikelihood_f_for_beta_hat = self._dlikelihood_f(beta_hat, self._np_freq_positive_reclassified, is_positive=True) + self._dlikelihood_f(beta_hat, self._np_freq_negative_reclassified, is_positive=False)
            dlikelihood_f_for_beta_hat_next = self._dlikelihood_f(beta_hat_next, self._np_freq_positive_reclassified, is_positive=True) + self._dlikelihood_f(beta_hat_next, self._np_freq_negative_reclassified, is_positive=False)
            delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T

            for i in range(self.exogen[0].size):
                delta_beta[i] = -dlikelihood_f_for_beta_hat_next[i] / (
                            dlikelihood_f_for_beta_hat_next[i] - dlikelihood_f_for_beta_hat[i]) * (
                                            beta_hat_next[i] - beta_hat[i])
            beta_hat = beta_hat_next
            beta_hat_next = beta_hat_next + delta_beta

        return beta_hat_next

    def fit_intercept(self):
        self.classify()
        self.reclassify(0.5)

        print("fitting.....")

        beta_hat = np.matrix(np.ones(self.exogen[0].size)).T
        beta_hat_next = np.matrix(np.zeros(self.exogen[0].size)).T

        while np.linalg.norm(self.full_cl_recl_dlikelihood_f(beta_hat_next)) >= self.METHOD_ACCURACY:
            dlikelihood_f_for_beta_hat_next = self.full_cl_recl_dlikelihood_f(beta_hat_next)
            delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T

            dlikelihood_derivative_approximation = np.zeros((self.exogen[0].size, self.exogen[0].size))

            for i in range(self.exogen[0].size):
                temp_beta = copy.deepcopy(beta_hat_next)
                temp_beta[i] = beta_hat[i]
                # FIXME: something bad with dimensions
                dlikelihood_derivative_approximation = (self.full_cl_recl_dlikelihood_f(beta_hat_next) - self.full_cl_recl_dlikelihood_f(temp_beta)) / (beta_hat_next[i] - beta_hat[i])

            delta_beta = - dlikelihood_f_for_beta_hat_next * np.linalg.inv(dlikelihood_derivative_approximation)
            beta_hat = beta_hat_next
            beta_hat_next = beta_hat_next + delta_beta

        return beta_hat_next

    def fit_without_classification(self):
        pass


def GEM(exogen_data, endogen_data, *args):
    return ApproximationGEMModelRedesigned(exogen_data, endogen_data)


def modulateRegression(regression_sample_quintity, regression_outlier_percentage):
    regressionParameters = ACCURATE_RESULT
    x_points = np.zeros(shape=[regression_sample_quintity, len(regressionParameters)])
    y_points = np.zeros(shape=regression_sample_quintity)

    for i in range(0, regression_sample_quintity):
        if random() > regression_outlier_percentage / 100:
            x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regressionParameters) - 1))
            y_points[i] = (x_points[i]*regressionParameters) + np.random.normal(0, 4)
        else:
            x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regressionParameters) - 1))
            y_points[i] = np.random.normal(100.0, 15.0, size=1)

    return x_points, y_points


x_points, y_points = modulateRegression(100, OUTLIER_PERCENTAGE)

APPROXIMATION_MODEL = GEM(x_points,y_points)
print(APPROXIMATION_MODEL.fit_intercept())

# APPROXIMATION_MODEL = sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
# tempHundredParams = APPROXIMATION_MODEL.fit().params

# APPROXIMATION_MODEL = sm.OLS(y_points,x_points, M=sm.robust.norms.HuberT())
# print(APPROXIMATION_MODEL.fit().params)
