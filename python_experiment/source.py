import numpy as np
import matplotlib.pyplot as plt
from random import random

SAMPLE_QUINTITY = 1000
OUTLIER_PERCENTAGE = 8.0
regressionParameters = np.matrix([90, 4]).T

x_points = np.zeros(shape=[SAMPLE_QUINTITY, len(regressionParameters)])
y_points = np.zeros(shape=SAMPLE_QUINTITY)

for i in range(0, SAMPLE_QUINTITY):
    if random() > OUTLIER_PERCENTAGE / 100:
        x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regressionParameters) - 1))
        # print(x_points[i])
        y_points[i] = (x_points[i] * regressionParameters) + np.random.normal(0, 4)
    else:
        x_points[i] = np.append(np.ones(1), np.random.uniform(-5, 5, size=len(regressionParameters) - 1))
        y_points[i] = np.random.normal(100, 15, size=1)


plt.plot(x_points.T[1], y_points, 'ro', markeredgecolor='k')
plt.xlabel("значение независимой переменной")
plt.ylabel("зависимая переменная")
plt.show()
