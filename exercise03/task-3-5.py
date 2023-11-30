import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from scipy.optimize import minimize
from numpy.random import multivariate_normal

def diffMatrix(u, v):
    return np.expand_dims(u,axis=1) - np.expand_dims(v,axis=0)

def prodMatrix(u, v):
    return np.expand_dims(u,axis=1) * np.expand_dims(v,axis=0)

def Kernel(xi, xj, theta1, theta2, theta3):
    return theta1 * np.exp(-((diffMatrix(xi, xj))**2)/(2*theta2**2)) + theta3 * prodMatrix(xi, xj) 
    
def KernelC( K, theta4):
    return K + theta4 * np.eye(K.shape[0])

#load data
data = np.loadtxt('whData.dat', dtype=np.object_, comments='#', delimiter=None)
w = data[:,0].astype(float)
h = data[:,1].astype(float)

#remove outliers
x = h[w!=-1]
y = w[w!=-1]

#optimized parameters from 3.4
theta_hat=[58.3175253, 12.5071223, 0.00001, 139.343116]
C = KernelC(Kernel(x, x, theta_hat[0], theta_hat[1], theta_hat[2]),theta_hat[3])

#Cholesky factorization
L = cholesky(C)
#sample w from normal distribution
w = multivariate_normal(np.zeros_like(x) , np.eye(len(x)))
y_dash = L @ w
#de-normalize
y_ = y_dash + np.mean(y)

#plot gt points and sampled points
plt.plot(x,y,'o',c='black', alpha=0.5)
plt.plot(x,y_,'o',c='r', alpha=0.5)
plt.xlim(145,205)
plt.ylim(35,115)
plt.show()
