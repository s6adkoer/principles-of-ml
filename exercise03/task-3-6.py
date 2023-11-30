import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv

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

#normalize y
y_hat= y-np.mean(y)

vecXs = np.linspace(140, 210, 200)
#optimal parameters from 3.4
theta_hat=[58.3175253, 12.5071223, 0.00001, 139.343116]

C = KernelC(Kernel(x, x, theta_hat[0], theta_hat[1], theta_hat[2]),theta_hat[3])

K_x_star = Kernel(x, vecXs, theta_hat[0], theta_hat[1], theta_hat[2])
K_star_x = Kernel(vecXs, x, theta_hat[0], theta_hat[1], theta_hat[2])
K_star_star = Kernel(vecXs, vecXs, theta_hat[0], theta_hat[1], theta_hat[2])

muh_dash = K_star_x @ inv(C) @ y_hat
Sigma_star = K_star_star - K_star_x @ inv(C) @ K_x_star
sigma_star = np.sqrt(np.diag(Sigma_star ))

#de-normalize
muh_star= muh_dash + np.mean(y)

#plot gt points(black) and GP prediction(red line) with cofidence interval
plt.plot(x,y,'o',c='black', alpha=0.5)
plt.plot(vecXs,muh_star ,c='r')
plt.fill_between(vecXs, muh_star-sigma_star, muh_star+sigma_star, color ='r', alpha=0.2)
plt.xlim(145,205)
plt.ylim(35,115)
plt.show()