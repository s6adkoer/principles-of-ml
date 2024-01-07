import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

### task 5.1.1
def phi(x, d):
    return [x**i for i in range(0,d+1)]

d=9

#load data
data=np.loadtxt('noisyCubicPoly.csv', dtype=np.float64, comments='#', delimiter=',')#.T
x=data[0]
y=data[1]
#calculate feature matrix
big_phi=np.array([ phi(x[i], d) for i in range(x.shape[0])]).T

### task 5.1.2

#numerical stable method for ordinary least squares
def least_squares(matX, vecY):
    Q,R=la.qr(matX.T)
    w_star = la.inv(R) @ Q.T@ vecY
    return w_star

w=least_squares(big_phi, y)

x2=np.linspace(-1,4,300)
big_phi2=np.array([ phi(x2[i], d) for i in range(x2.shape[0])]).T

#plot training data and fitted model
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)

ax1.scatter(x,y)
ax1.plot(x2,big_phi2.T @ w)
    
ax1.set_ylim(-6,1)
ax1.set_xlim(-1,4)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines['left'].set_position(('data',0))
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['left'].set_color('gray')
ax1.spines['bottom'].set_color('gray')

plt.show()

### task 5.1.3


def reg_least_squares(matX, lambda_value ,vecY):
    Q,R=la.qr(matX.T )
    w_star = la.inv(R.T @ R + lambda_value * np.eye(R.shape[1])) @R.T@ Q.T@ vecY 
    
    return w_star

lambda_V= [0.5, 0.005, 5]

for l in lambda_V:

    print('lamda = ', l)
    w=reg_least_squares(big_phi,l,  y)

    x2=np.linspace(-1,4,300)
    big_phi2=np.array([ phi(x2[i], d) for i in range(x2.shape[0])]).T


    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)

    ax1.scatter(x,y)
    ax1.plot(x2,big_phi2.T @ w)

    ax1.set_ylim(-6,2)
    ax1.set_xlim(-1,4)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines['left'].set_position(('data',0))
    ax1.spines['bottom'].set_position(('data',0))
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')

    plt.show()
