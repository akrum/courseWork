import numpy as np
from random import random
import math
from multiprocessing import Process, Queue , Pipe
import threading
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
accurate_result=np.matrix([90,4,5]).T
epsilon = 8.0
class approximationGEMModel:
    def __init__(self, exogen_data, endogen_data):
        self.a = 8.0/(3.0*math.pi)*(3.0-math.pi)/(math.pi-4.0)
        self.interval_length = 0.001
        self.k = 100000000
        self.intervals_left_bound = 0.0-self.interval_length*self.k/2.0
        self.intervals_right_bound = 0.0+self.interval_length*self.k/2.0
        self.exogen=exogen_data
        self.endogen=endogen_data
        self.sigmasq= 225.0
        self.threadingEvent = threading.Event()
    def erf(self, value):
        # print "{0:f}\n".format(math.sqrt(1.0-math.exp((-1.0*value*value)*(4.0/math.pi+self.a*value*value)/(1.0+self.a*value*value))))
        return math.sqrt(1.0-math.exp((-1.0*value*value)*(4.0/math.pi+self.a*value*value)/(1.0+self.a*value*value)))
        
        # return scipy.special.erf(value)
    def derf(self, value):
        temp=math.exp((-1*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value))
        temp*=((-2*value+2*self.a*value*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value)-2*self.a*value*value*value/(1+self.a*value*value))
        temp/= 2*math.sqrt(1-math.exp((-1*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value)))
        return temp
    def P(self, x_i, y_i, mu_i, beta_hat):
        if mu_i==0:
            return 0.5*(1+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq))))
        if mu_i==self.k-1:
            return 0.5*(1+self.erf((self.intervals_right_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq))))
        a_mu_i_plus_1 = (mu_i-self.k/2)*self.interval_length
        a_mu_i = (mu_i-self.k/2)*self.interval_length-self.interval_length
        return 0.5*(self.erf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))-self.erf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq))))
    def dP(self, x_i, y_i, mu_i, beta_hat):
        if mu_i==0:
            return 2.0*x_i*self.derf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))).item((0,0))
        if mu_i==self.k-1:
            return 2.0*x_i*self.derf((self.intervals_right_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))).item((0,0))
        a_mu_i_plus_1 = (mu_i-self.k/2)*self.interval_length
        a_mu_i = (mu_i-self.k/2)*self.interval_length-self.interval_length
        if self.erf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))-self.erf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))==0.0:
            return 0
        temp = 0.0
        temp+= self.derf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))-self.derf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))
        temp/= self.erf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))-self.erf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0*self.sigmasq)))
        return 2.0*x_i*temp.item((0,0))
    
    def likelihood_f(self, beta, mu_data):
        current_likelihood_result = 0.0
        for i in range(0, self.endogen.size):
            current_likelihood_result+= np.log(self.P(self.exogen[i], self.endogen[i], mu_data[i], beta))
        return current_likelihood_result
    def dlikelihood_f(self, beta, mu_data):
        current_likelihood_derivative_result = np.zeros(self.exogen[0].size)
        for i in range(0, self.endogen.size):
            current_likelihood_derivative_result+= self.dP(self.exogen[i], self.endogen[i], mu_data[i], beta)
        # print current_likelihood_derivative_result
        return -0.5*current_likelihood_derivative_result
    # def classificate(self):
    #     self.mu_data = np.zeros(self.endogen.size)
    #     for i in range(0, self.endogen.size):
    #         self.mu_data[i] = int(round(self.endogen[i]/self.interval_length))
    #     # print (self.endogen[0],self.mu_data[0])
    #     # print int(round(-2.3/self.interval_length))
    def classificate(self):
        self.mu_data = np.zeros(self.endogen.size)
        for i in range(self.endogen.size):
            if self.endogen[i]<self.intervals_left_bound:
                self.mu_data[i]=0
            elif self.endogen[i]>self.intervals_right_bound:
                self.mu_data[i]=self.k-1
            else:
                self.mu_data[i] = int(round(self.endogen[i]/self.interval_length))+self.k/2
        print("classified")
        return self
    def reclassificate(self, delta):
        self.mu_data_reclassificated = np.zeros(self.endogen.size)
        for i in range(0, self.endogen.size):
            current_faced_classes={}
            for j in range(0, self.endogen.size):
                # print (np.linalg.norm(self.exogen[i]-self.exogen[j]))
                if  (np.linalg.norm(self.exogen[i]-self.exogen[j])<=delta) :
                    if self.mu_data[j] in current_faced_classes:
                        current_faced_classes[self.mu_data[j]]+=1
                    else:
                        current_faced_classes[self.mu_data[j]]=1
            maximumfacedtimes = 1
            maximumfacedclass = self.mu_data[i]
            for key in current_faced_classes:
                # print (key, current_faced_classes[key])
                if maximumfacedtimes < current_faced_classes[key]:
                    maximumfacedtimes = current_faced_classes[key]
                    maximumfacedclass = key
            # print (maximumfacedtimes, maximumfacedclass)
            self.mu_data_reclassificated[i] = maximumfacedclass
        print("reclassified")
        return self
    def dichotomy_var(self, beta_0, beta_1, possible_beta_hats, psi, variable_index):
        if abs(beta_0[variable_index]-beta_1[variable_index])<=2*psi[variable_index]:
            possible_beta_hats.append(beta_0)
            return
        beta_2 = 0.5*(beta_0+beta_1)
        
        if self.dlikelihood_f(beta_2)[variable_index]*self.dlikelihood_f(beta_0)[variable_index]<0:
            self.dichotomy_var(beta_2,beta_0,possible_beta_hats, psi, variable_index)
        if self.dlikelihood_f(beta_2)[variable_index]*self.dlikelihood_f(beta_1)[variable_index]<0:
            self.dichotomy_var(beta_2,beta_1,possible_beta_hats, psi, variable_index)    
        
    def dichotomy(self, beta_0, beta_1, possible_beta_hats, psi):
        # for i in range(start_index, beta_0.size):
        #     temp_possible_beta_hats = []
        #     self.dichotomy_var(beta_0, beta_1, temp_possible_beta_hats, psi, i)
        #     for item in temp_possible_beta_hats:
        #         tempbeta_0 = np.matrix(beta_0)
        #         tempbeta_1 = np.matrix(beta_1)
        #         tempbeta_0[i]=item
        #         tempbeta_1[i]=item
        #         dichotomy

        tempComparisonResult = np.less(self.dlikelihood_f(beta_0).T*self.dlikelihood_f(beta_1),np.zeros(self.exogen[0].size).T)
        print(tempComparisonResult)
        if(np.less(np.abs(beta_0-beta_1),2*psi)[0].all()):
            possible_beta_hats.append(beta_0)
            return
        beta_2 = 0.5*(beta_0+beta_1)
        # tempComparisonResult = np.less(self.dlikelihood_f(beta_1).T*self.dlikelihood_f(beta_0),np.zeros(self.exogen[0].size).T)
        # print tempComparisonResult
        # for i in range(0, tempComparisonResult.size):
        #     if tempComparisonResult[i]==True:
        #         beta_2[i]= 0.5*(beta_0[i]+beta_1[i])
        # del tempComparisonResult beta_2 = np.matrix(beta_0)
        
        tempComparisonResult = np.less(self.dlikelihood_f(beta_2).T*self.dlikelihood_f(beta_0),np.zeros(self.exogen[0].size).T)
        if tempComparisonResult[0].all():
            self.dichotomy(beta_2,beta_0,possible_beta_hats,psi)
        
        del tempComparisonResult
        tempComparisonResult = np.less(self.dlikelihood_f(beta_2).T*self.dlikelihood_f(beta_1),np.zeros(self.exogen[0].size).T)
        if tempComparisonResult[0].all():
            self.dichotomy(beta_2,beta_1,possible_beta_hats,psi)
        
    def gradient(self, beta_0,possible_betas):
        nabla_beta = self.dlikelihood_f(beta_0).reshape((beta_0.size,1))
        beta_new = beta_0-nabla_beta
        # print beta_new
        if np.less(np.abs(beta_0-beta_new),np.full(beta_0.size,1e-1)).all():
            possible_betas.append(beta_new)
            self.threadingEvent.set()
            return 0
        else:
            gradientThread = threading.Thread(target=self.gradient, args=(beta_new, possible_betas))
            gradientThread.start()
    def fit(self):
        self.classificate()
        self.reclassificate(0.5)

        print("fitting.....")
        beta_hat=np.matrix(np.ones(self.exogen[0].size)).T
        # print self.dlikelihood_f(beta_hat, self.mu_data)
        beta_hat_next = np.matrix(np.zeros(self.exogen[0].size)).T

        while(np.linalg.norm(self.dlikelihood_f(beta_hat_next,self.mu_data_reclassificated))>=0.1):
            dlikelihood_f_for_beta_hat = self.dlikelihood_f(beta_hat, self.mu_data_reclassificated)
            dlikelihood_f_for_beta_hat_next = self.dlikelihood_f(beta_hat_next, self.mu_data_reclassificated)
            delta_beta = np.matrix(np.zeros(self.exogen[0].size)).T
            for i in range(self.exogen[0].size):
                delta_beta[i] = -dlikelihood_f_for_beta_hat_next[i]/(dlikelihood_f_for_beta_hat_next[i]-dlikelihood_f_for_beta_hat[i])*(beta_hat_next[i]-beta_hat[i])
            beta_hat = beta_hat_next
            beta_hat_next = beta_hat_next+delta_beta



        # withoud_classification = self.dlikelihood_f(accurate_result, self.mu_data)
        # print withoud_classification

        # temp_delta = 1.0
        # self.reclassificate(temp_delta)
        # temp_accurate_result = self.dlikelihood_f(accurate_result, self.mu_data_reclassificated)
        # while np.equal(temp_accurate_result,accurate_result).all():
        #     temp_delta+=1.0
        #     temp_accurate_result = self.dlikelihood_f(accurate_result, self.mu_data_reclassificated)
        # print self.dlikelihood_f(accurate_result, self.mu_data_reclassificated)
        return beta_hat_next
    def compare(self):
        self.classificate()
        beta_hat=np.matrix([170,8]).T
        print(self.dlikelihood_f(beta_hat, self.mu_data))
        without_classification = self.dlikelihood_f(accurate_result, self.mu_data)
        print(without_classification)

        temp_delta = 1.0
        self.reclassificate(temp_delta)
        with_classification = self.dlikelihood_f(accurate_result, self.mu_data_reclassificated)
        while np.equal(with_classification,accurate_result).all():
            temp_delta+=0.5
            with_classification = self.dlikelihood_f(accurate_result, self.mu_data_reclassificated)
    
        print(self.dlikelihood_f(accurate_result, self.mu_data_reclassificated))
        return (without_classification, with_classification)
def GEM(exogen_data, endogen_data, *args):
    return approximationGEMModel(exogen_data,endogen_data)




def modulateRegression(regressionSampleQuintity,regressionOutlierPercentage):
    regressionParameters = accurate_result 
    x_points = np.zeros(shape=[regressionSampleQuintity,len(regressionParameters)])
    y_points = np.zeros(shape = regressionSampleQuintity)
    for i in range(0,regressionSampleQuintity):
        if random()>regressionOutlierPercentage/100:
            x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
            y_points[i]=(x_points[i]*regressionParameters)+np.random.normal(0,4)
        else:
            x_points[i] = np.append(np.ones(1),np.random.uniform(-5,5,size = len(regressionParameters)-1))
            y_points[i]=np.random.normal(100.0,15.0, size=1)
    return (x_points,y_points)

x_points,y_points=modulateRegression(1000, epsilon)
# epsilons = np.zeros(1)
# not_classificated = np.zeros(1)
# classificated = np.zeros(1)
# while epsilon<=52.0:
#     print "testing with epsilon {0:f}".format(epsilon)
#     epsilons=np.append(epsilons,epsilon)
#     x_points,y_points=modulateRegression(1000, epsilon)
#     appro_model = GEM(x_points,y_points)
#     without_class, with_class = appro_model.compare()
#     not_classificated=np.append(not_classificated,np.linalg.norm(without_class))
#     classificated= np.append(classificated, np.linalg.norm(with_class))
    # epsilon+=2.0
    # print "\n"
# lines = plt.plot(epsilons,not_classificated,'r', epsilons,classificated,'b', label="...")
# plt.legend(lines, ["not classified","classified", "blabla"], loc="lower right")
# plt.title("likelyhood derivatives")
# ax = plt.gca()
# plt.show()
APPROXIMATION_MODEL = GEM(x_points,y_points)
print(APPROXIMATION_MODEL.fit())
APPROXIMATION_MODEL=sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
tempHundredParams=APPROXIMATION_MODEL.fit().params
# print(tempHundredParams)