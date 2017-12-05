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
#breakpoint search
accurate_result=np.matrix([100,4,5,9,7]).T
def modulateRegression(regressionSampleQuintity,regressionOutlierPercentage):
    regressionParameters = accurate_result #надо брать свободный член больше
    x_points = np.zeros(shape=[regressionSampleQuintity,len(regressionParameters)])
    y_points = np.zeros(shape = regressionSampleQuintity)
    # plt.plot(x_points,y_points,'ro')
    # # plt.hist(y_points,bins="auto")
    # plt.show()
    for i in range(0,regressionSampleQuintity):
        if random()>regressionOutlierPercentage/100:
            x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
            # print(x_points[i])
            y_points[i]=(x_points[i]*regressionParameters)+np.random.normal(0,4)
        else:
            x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
            y_points[i]=np.random.normal(100,10, size=1)
    return (x_points,y_points)
cycleOulierPercentage = 5
while cycleOulierPercentage<100:
    print("Going to perform test with percentage {0:d}%....".format(cycleOulierPercentage))
    x_points,y_points=modulateRegression(100,cycleOulierPercentage)
    APPROXIMATION_MODEL=sm.RLM(y_points,x_points,M=sm.robust.norms.HuberT())
    tempHundredParams=APPROXIMATION_MODEL.fit().params
    print(tempHundredParams)

    x_points,y_points=modulateRegression(3000,cycleOulierPercentage)
    APPROXIMATION_MODEL = sm.RLM(y_points,x_points,M=sm.robust.norms.HuberT())
    tempThousandParams = APPROXIMATION_MODEL.fit().params
    print(tempThousandParams)
    if np.linalg.norm(accurate_result-tempHundredParams)<=np.linalg.norm(accurate_result-tempThousandParams):
        print("Breakpoint for this approximation model is:{0:d}%".format(cycleOulierPercentage))
        break
    cycleOulierPercentage +=5

