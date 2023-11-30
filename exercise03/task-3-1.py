import numpy as np

def diffMatrix(u,v):
    return np.subtract.outer(u,v)

def prodMatrix(u,v):
    return np.multiply.outer(u,v)

print("Task-3-1-1", diffMatrix(np.array([1,2,3]),np.array([10,9])))
print("Task-3-1-2", prodMatrix(np.array([1,2,3]),np.array([10,9])))