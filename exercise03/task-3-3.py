from numpy.random import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


def diffMatrix(u, v):
    return np.expand_dims(u,axis=1) - np.expand_dims(v,axis=0)
    
def prodMatrix(u, v):
    return np.expand_dims(u,axis=1) * np.expand_dims(v,axis=0)

def linearKernelMatrix(u, v, alpha):
    return alpha * prodMatrix(u, v)


def gaussKernelMatrix(u,v, alpha, sigma):
    return alpha * np.exp(-((diffMatrix(u, v))**2)/(2*sigma**2))

vecX = np.linspace(-5.0, 15.0, 55)
vec0 = np.zeros_like(vecX)

####
# task 3.3.1
####

#create figure

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

# create 5 samples from the linearKernelMatrix

for i in range(5):
    y=multivariate_normal(vec0 , linearKernelMatrix(vecX,vecX, 1))
    ax1.plot(vecX, y,'o-', alpha=0.4)

#set figure attributes
    
ax1.set_ylim(-10,10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines['left'].set_position(('data',0))
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['left'].set_color('gray')
ax1.spines['bottom'].set_color('gray')

plt.show()


####
# task 3.3.2
####

#create figure

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

# create 5 samples from the gaussKernelMatrix

for i in range(5):
    y=multivariate_normal(vec0 , gaussKernelMatrix(vecX,vecX, alpha=6, sigma=1.5))
    ax1.plot(vecX, y,'o-', alpha=0.4)

#set figure attributes
    
ax1.set_ylim(-10,10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines['left'].set_position(('data',0))
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['left'].set_color('gray')
ax1.spines['bottom'].set_color('gray')

plt.show()

####
# task 3.3.3
####

#create figure

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

# create 5 samples from the sum of linearKernelMatrix and gaussKernelMatrix

for i in range(5):
    y=multivariate_normal(vec0 , linearKernelMatrix(vecX,vecX, 2)+gaussKernelMatrix(vecX,vecX, alpha=6, sigma=1.5))
    ax1.plot(vecX, y,'o-', alpha=0.4)

#set figure attributes
    
ax1.set_ylim(-10,10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines['left'].set_position(('data',0))
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['left'].set_color('gray')
ax1.spines['bottom'].set_color('gray')

plt.show()
