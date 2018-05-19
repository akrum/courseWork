import numpy as np
from random import random
import math
accurate_result=np.matrix([100,4]).T
epsilon = 0.01
class approximationGEMModel:
    def erf(self, value):
        # print "{0:f}\n".format(math.sqrt(1.0-math.exp((-1.0*value*value)*(4.0/math.pi+self.a*value*value)/(1.0+self.a*value*value))))
        return math.sqrt(1.0-math.exp((-1.0*value*value)*(4.0/math.pi+self.a*value*value)/(1.0+self.a*value*value)))
    def derf(self, value):
        temp=math.exp((-1*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value))
        temp*=((-2*value+2*self.a*value*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value)-2*self.a*value*value*value/(1+self.a*value*value))
        temp/= 2*math.sqrt(1-math.exp((-1*value*value)*(4/math.pi+self.a*value*value)/(1+self.a*value*value)))
        return temp
    def dP(self, x_i, y_i, mu_i, beta_hat):
        if mu_i==0:
            return self.derf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2)*self.sigmasq))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2)*self.sigmasq)))
        if mu_i==self.k-1:
            return self.derf((self.intervals_right_bound-x_i*beta_hat)/(math.sqrt(2)*self.sigmasq))/(1.0+self.erf((self.intervals_left_bound-x_i*beta_hat)/(math.sqrt(2)*self.sigmasq)))
        a_mu_i_plus_1 = mu_i/2*self.interval_length+self.interval_length
        a_mu_i = mu_i/2*self.interval_length+0.0
        if self.erf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))-self.erf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))==0.0:
            return 0
        temp = 0.0
        temp+= self.derf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))-self.derf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))
        temp/= self.erf((a_mu_i_plus_1-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))-self.erf((a_mu_i-x_i*beta_hat)/(math.sqrt(2.0)*self.sigmasq))
        return x_i*temp.item((0,0))
    def dlikelihood_f(self, beta):
        current_likelihood_derivative_result = 0.0
        for i in range(0, self.endogen.size):
            current_likelihood_derivative_result+= self.dP(self.exogen[i], self.endogen[i], self.mu_data[i], beta)
        # print current_likelihood_derivative_result
        return current_likelihood_derivative_result
    def __init__(self, exogen_data, endogen_data):
        self.a = 8.0/(3.0*math.pi)*(3.0-math.pi)/(math.pi-4.0)
        self.interval_length = 1.0
        self.k = 100000
        self.intervals_left_bound = 0.0-self.interval_length*self.k/2.0
        self.intervals_right_bound = 0.0+self.interval_length*self.k/2.0
        self.exogen=exogen_data
        self.endogen=endogen_data
        self.sigmasq= 16.0
    def classificate(self):
        self.mu_data = np.zeros(self.endogen.size)
        for i in range(0, self.endogen.size):
            self.mu_data[i] = int(round(self.endogen[i]/self.interval_length))
        # print int(round(-2.3/self.interval_length))
    def reclassificate(self):
        return 0
    def dichotomy(self, beta_0, beta_1, possible_beta_hats, psi):
        tempComparisonResult = np.less(self.dlikelihood_f(beta_0).T*self.dlikelihood_f(beta_1),np.zeros(self.exogen[0].size).T)
        print tempComparisonResult
        # assert tempComparisonResult[0].all()
        if(np.less(np.abs(beta_0-beta_1),2*psi)[0].all()):
            possible_beta_hats.append(beta_0)
            return
        beta_2 = 0.5*(beta_0+beta_1)
        # tempComparisonResult = np.less(self.dlikelihood_f(beta_1).T*self.dlikelihood_f(beta_0),np.zeros(self.exogen[0].size).T)
        # for i in range(0, tempComparisonResult[0].size):
        #     if tempComparisonResult[0][i]==False:
        #         beta_2[i]= 0.5*(beta_0[i]+beta_1[i])
        
        del tempComparisonResult
        tempComparisonResult = np.less(self.dlikelihood_f(beta_2).T*self.dlikelihood_f(beta_0),np.zeros(self.exogen[0].size).T)
        if tempComparisonResult[0].any():
            self.dichotomy(beta_2,beta_0,possible_beta_hats,psi)
        
        del tempComparisonResult
        tempComparisonResult = np.less(self.dlikelihood_f(beta_2).T*self.dlikelihood_f(beta_1),np.zeros(self.exogen[0].size).T)
        if tempComparisonResult[0].any():
            self.dichotomy(beta_2,beta_1,possible_beta_hats,psi)
        
        
        
    def fit(self):
        self.classificate()
        beta_hat=np.matrix(np.ones(self.exogen[0].size)).T
        print self.dlikelihood_f(beta_hat)
        print self.dlikelihood_f(accurate_result)
        dichotomyPossibleBetaHats = []
        self.dichotomy(np.matrix(np.full(self.exogen[0].size, -200)).T,np.matrix(np.full(self.exogen[0].size, 200)).T,dichotomyPossibleBetaHats, np.full(self.exogen[0].size, 0.0001))
        print(dichotomyPossibleBetaHats)
        return np.zeros(self.exogen.size)
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
            y_points[i]=np.random.normal(100,10, size=1)
    return (x_points,y_points)

x_points,y_points=modulateRegression(100, 1.0)
APPROXIMATION_MODEL = GEM(x_points,y_points)
print APPROXIMATION_MODEL.fit()
# APPROXIMATION_MODEL=sm.RLM(y_points,x_points, M=sm.robust.norms.HuberT())
# tempHundredParams=APPROXIMATION_MODEL.fit().params