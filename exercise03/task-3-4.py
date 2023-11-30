import numpy as np
from numpy.linalg import det, inv
from scipy.optimize import minimize

def diffMatrix(u, v):
    return np.expand_dims(u,axis=1) - np.expand_dims(v,axis=0)

def prodMatrix(u, v):
    return np.expand_dims(u,axis=1) * np.expand_dims(v,axis=0)

#load data
data = np.loadtxt('whData.dat', dtype=np.object_, comments='#', delimiter=None)
w = data[:,0].astype(float)
h = data[:,1].astype(float)
#remove outliers
x = h[w!=-1]
y = w[w!=-1]
#normalize y
y=y-np.mean(y)

def Kernel(xi, xj, theta1, theta2, theta3):
    return theta1 * np.exp(-((diffMatrix(xi, xj))**2)/(2*theta2**2)) + theta3 * prodMatrix(xi, xj) 
    
def KernelC( K, theta4):
    return K + theta4 * np.eye(K.shape[0])
        
def negLikelihood(theta, x, y_dash):
    C= KernelC(Kernel(x, x, theta[0], theta[1], theta[2]),theta[3])
    return 1/2 * np.log(det(C)) + 1/2 * y_dash.T @ inv(C) @ y_dash

#optimal parameters should not be negative
bounds = [(1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None)]
res = minimize(negLikelihood,np.array([1.0, 20.0, 0.5, 1.0]), args=(x,y), bounds = bounds)
print(res.x)