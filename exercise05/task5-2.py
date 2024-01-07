import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#load data
data=np.loadtxt('noisyCubicPoly.csv', dtype=np.float64, comments='#', delimiter=',')
x=data[0]
y=data[1]

def kernelMatrix(x,d,b):
    return (b+ x[...,None]@ x[None,...])**d

def kernelVector(x_trn, x_tst ,d ,b ):
    return (b+ x_trn[None,...].T@ x_tst[None,...])**d
   
#calculate  polynomial kernel matrix and  polynomial kernel vector
K=kernelMatrix(x,d=3,b=1)
k=kernelVector(x,x,d=3,b=1)

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

k=kernelVector(x,np.linspace(-1,4,300),d=3,b=1)
#compute prediction
prediction=k.T@ la.inv(K + 0.5 * np.eye(data.shape[1])) @ y
#plot prediction
ax1.plot(np.linspace(-1,4,300),prediction,color='red', label='prediction')


#plot gt
ax1.scatter(x,y,color='black', label='gt')
    
ax1.set_ylim(-6,1)
ax1.set_xlim(-1,4)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines['left'].set_position(('data',0))
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['left'].set_color('gray')
ax1.spines['bottom'].set_color('gray')
ax1.legend()

plt.show()