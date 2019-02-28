import math
import numpy as np


class CreateDefines:
    METHOD_ACCURACY = 1e-7
    INTERVAL_LENGTH = 1e-1
    K_EVERY_SEGMENT = 50000000
    SIGMA_SQ = 225.0
    THREAD_JOIN_TIMEOUT = 10
    COUNT_LIMIT_OPERATIONS = 1e2
    LEFT_BOUND_EVERY_VAR_INDENT = 20.0
    RECLASSIFICATION_LEVEL = 10

    @property
    def a(self):
        return 8.0 / (3.0 * math.pi) * (3.0 - math.pi) / (math.pi - 4.0)

    @property
    def intervals_left_bound(self):
        return 0.0 - self.INTERVAL_LENGTH * self.K_EVERY_SEGMENT / 2.0

    @property
    def intervals_right_bound(self):
        return 0.0 + self.INTERVAL_LENGTH * self.K_EVERY_SEGMENT / 2.0

    @staticmethod
    def right_bound_fit_indent(size):
        return np.matrix([10.0 for _ in range(size)]).T

    @staticmethod
    def fit_loop_stop_value(size):
        return np.matrix([100.0 for _ in range(size)]).T

    @staticmethod
    def left_bound_fit_init(size):
        return np.matrix([-20.0 for _ in range(size)]).T


GroupingEstimatesDefines = CreateDefines()
