import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


#load data
data=np.loadtxt('noisyCubicPoly.csv', dtype=np.float64, comments='#', delimiter=',')#.T
x=data[0]
y=data[1]

#initialize Parameter 
C=2
b=1
d=9

def kernelMatrix(x,d,b):
    return (b+ x[...,None]@ x[None,...])**d

def kernelVector(x_trn, x_tst ,d ,b ):
    return (b+ x_trn[None,...].T@ x_tst[None,...])**d
   
K=kernelMatrix(x,d,b)


#calculate lambda_hat and b_hat
helpi=np.vstack((np.hstack((K+ 1/C * np.eye(K.shape[0]), np.ones((1,11)).T)),np.hstack((np.ones(11), np.array([0])))))
helpi2=la.inv(helpi) @ np.hstack((y, np.array([0])))
lambda_hat, b_hat =helpi2[:11],helpi2[-1]


fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

ax1.scatter(x,y, color='black',label='gt')

#calculate prediction and plot it
k=kernelVector(x,np.linspace(-1,4,300),d,b)
y_hat=k.T@ lambda_hat + b_hat
ax1.plot(np.linspace(-1,4,300),y_hat,color='red', label='prediction')

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