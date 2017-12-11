import numpy as np
import matplotlib.pyplot as plt
from random import random
import pylab
import scipy
from outliers import smirnov_grubbs as grubbs
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.robust.scale import mad
import theano
import theano.tensor as T
import statsmodels.api as sm
import statsmodels.formula.api as smf

SAMPLE_QUINTITY=1000
OUTLIER_PERCENTAGE = 10.0
regressionParameters = np.matrix([100,4]).T 


x_points = np.zeros(shape=[SAMPLE_QUINTITY,len(regressionParameters)])
y_points = np.zeros(shape = SAMPLE_QUINTITY)
# plt.plot(x_points,y_points,'ro')
# # plt.hist(y_points,bins="auto")
# plt.show()
for i in range(0,SAMPLE_QUINTITY):
    if random()>OUTLIER_PERCENTAGE/100:
        x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
        # print(x_points[i])
        y_points[i]=(x_points[i]*regressionParameters)+np.random.normal(0,4)
    else:
        x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
        y_points[i]=np.random.normal(100,10, size=1)
# plt.hist(y_points, bins="auto")
# plt.show()
# plt.plot(x_points.T[1],y_points,'ro')
# plt.show()


# modelspec = ('cost ~ np.log(units) + np.log(units):item + item') #where item is a categorical variable
# results = smf.rlm(modelspec, data = y_points, M = sm.robust.norms.TukeyBiweight()).fit()
# print results.summary()


rlm_model = sm.RLM(y_points, x_points, M=sm.robust.norms.HuberT())
ols_model = sm.OLS(y_points,x_points)
rec_model = sm.RecursiveLS(y_points,x_points)
rlm_results = rlm_model.fit()
ols_results = ols_model.fit()
rec_results = rec_model.fit()
#http://www.statsmodels.org/stable/examples/notebooks/generated/recursive_ls.html
print(ols_results.params)
print(rlm_results.params)
print(rec_results.params)

#OLS and recursiveLS have the same results
print(rlm_results.summary())

