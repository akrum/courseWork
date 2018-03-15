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
from multiprocessing import Process, Queue , Pipe
#breakpoint search
accurate_result=np.matrix([100,4]).T
N_MonteCarlo = 20
epsilon = 0.01
deltasHundred = np.zeros(1)
deltasThousand = np.zeros(1)
epsilons=np.zeros(1)
def modulateRegression(regressionSampleQuintity,regressionOutlierPercentage):
    regressionParameters = accurate_result 
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
cycleOulierPercentage = 1.0
def rlmForHundred(conn, cycleOulierPercentage, hundredAndThousandConn_send):
    x_points,y_points=modulateRegression(1000,cycleOulierPercentage)
    # hundredAndThousandConn_send.send(x_points)
    # hundredAndThousandConn_send.send(y_points)
    APPROXIMATION_MODEL=sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
    tempHundredParams=APPROXIMATION_MODEL.fit().params
    conn.send(tempHundredParams)
def rlmForThousand(conn, cycleOulierPercentage, hundredAndThousandConn_recv):
    # x_points = hundredAndThousandConn_recv.recv()
    # y_points = hundredAndThousandConn_recv.recv()
    # temp_x_points,temp_y_points=modulateRegression(2000,cycleOulierPercentage)
    # x_points = np.append(x_points, temp_x_points)
    # y_points = np.append(y_points, temp_y_points)
    # plt.plot(x_points,y_points,'ro')
    # plt.show()
    x_points,y_points=modulateRegression(2000,cycleOulierPercentage)
    APPROXIMATION_MODEL = sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
    tempThousandParams = APPROXIMATION_MODEL.fit().params
    conn.send(tempThousandParams)
while cycleOulierPercentage<100:
    print("Going to perform test with percentage {0:f}%....".format(cycleOulierPercentage))
    discrepancyHundred=0.0
    discrepancyThousand=0.0
    for i in range(1,int(N_MonteCarlo+1)):
        parent_conn_hundred, child_conn_hundred = Pipe()
        parent_conn_thousand, child_conn_thousand = Pipe()
        hundredAndThousandConn_recv, hundredAndThousandConn_send = Pipe()
        pHundred = Process(target=rlmForHundred, args=(child_conn_hundred, cycleOulierPercentage, hundredAndThousandConn_send))
        pThousand = Process(target=rlmForThousand, args=(child_conn_thousand,cycleOulierPercentage, hundredAndThousandConn_recv))
        pHundred.start()
        pThousand.start()
        # x_points,y_points=modulateRegression(1000,cycleOulierPercentage)
        # APPROXIMATION_MODEL=sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
        tempHundredParams= parent_conn_hundred.recv()
        tempThousandParams = parent_conn_thousand.recv()
        discrepancyHundred += np.linalg.norm(np.squeeze(np.asarray(accurate_result.T-tempHundredParams)))
        # x_points,y_points=modulateRegression(3000,cycleOulierPercentage)
        # APPROXIMATION_MODEL = sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
        # tempThousandParams = APPROXIMATION_MODEL.fit().params
        discrepancyThousand += np.linalg.norm(np.squeeze(np.asarray(accurate_result.T-tempThousandParams)))
        # pHundred.join()
        # pThousand.join()
        # plt.plot(x_points.T[1],y_points,'ro')
        # plt.show()
        pHundred.terminate()
        pThousand.terminate()
    discrepancyHundred/=N_MonteCarlo
    discrepancyThousand/=N_MonteCarlo
    deltasHundred = np.append(deltasHundred, discrepancyHundred)
    deltasThousand =  np.append(deltasThousand, discrepancyThousand)
    epsilons = np.append(epsilons,cycleOulierPercentage)
    print("Disperancies: {0:f}, {1:f}".format(discrepancyHundred,discrepancyThousand))
    if discrepancyHundred<=discrepancyThousand:
        print("Breakpoint for this approximation model is:{0:f}%".format(cycleOulierPercentage))
        plt.plot(epsilons,deltasHundred,'r', epsilons,deltasThousand,'b')
        plt.show()
        break
    # plt.plot(epsilons,deltasHundred,'ro')
    # plt.show()
    cycleOulierPercentage +=1.0

