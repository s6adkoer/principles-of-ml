import numpy as np

# we can't import task 3-1 easily because of the file name ...
# Therefore we solve the task again, this time using broadcasting

def diffMatrix(u,v):
    return u[:,np.newaxis] - v

def prodMatrix(u,v):
    return u[:,np.newaxis] * v

def linearKernelMatrix(u,v,alpha):
    return alpha*prodMatrix(u,v)

def gaussKernelMatrix(u,v,alpha,sigma):
    return alpha*(np.exp(-(u[:,np.newaxis] - v)**2/(2*sigma**2)))
u=np.array([1,2,3])
v=np.array([10,9])
print("Task 3-2-1",linearKernelMatrix(np.array([1,2,3]),np.array([10,9]),1))
print("Task 3-2-2",gaussKernelMatrix(np.array([1,2,3]),np.array([5,3]),3,1))
print(np.expand_dims(u,axis=1) * np.expand_dims(v,axis=0))
print(prodMatrix(u,v))