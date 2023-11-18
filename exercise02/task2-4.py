import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

@np.vectorize
def F(t,alpha,beta):
    """Calculate the probability of X\in (-\infty, t] for a given weibull distribution"""
    return 1-np.exp(-(t/beta)**alpha)

@np.vectorize
def f(t,alpha,beta):
    """Calculate the density at point t for a given weibull distribution"""
    return (alpha/beta)*np.power((t/beta),(alpha-1))*np.exp(-np.power((t/beta),alpha))

def get_pi(i,ts,alpha,beta): 
    if i>0:
        return F(ts[i],alpha,beta)-F(ts[i-1],alpha,beta) # Base case
    elif i==0:
        return F(ts[i],alpha,beta) # The probability of landing in $(-\infty, t_0]$ which is $F(t_0)-\lim_{x\to \infty} F(x)=F(t_0)-0$
    else:
        raise ValueError

def get_residuals(N,hs,ts,alpha,beta): 
    """Calculate residuals needed by scipy.least_squares"""
    ps=np.array([get_pi(i,ts,alpha,beta) for i in range(ts.shape[0])])
    ws=1/(N*ps)
    return (hs-N*ps)*np.sqrt(ws)

data = data=np.loadtxt("myspace.csv",dtype=object,delimiter=",")

time_frame = data[:,0].astype(str)
global_interest = data[:,1].astype(float)

to_skip=np.where(global_interest != 0) 
ts=np.array([1+i for i in range(np.array(to_skip).shape[1])])
hs=global_interest[to_skip]

N=np.sum(hs)

theta_hat=least_squares(lambda theta:get_residuals(N,hs,ts,*theta),[1,100],bounds=([0,0],np.inf),ftol=1e-10).x

f_tilde= lambda t: f(t,*theta_hat)

print(theta_hat)

plt.plot(ts,hs,label="Myspace data",c="gray")
plt.plot(ts,f_tilde(ts)*np.sum(hs),label="Fitted Weibull",c="red")
plt.xlabel("Timestep")
plt.ylabel("Global Interest")
plt.legend()
plt.gca().set_aspect(1)
plt.savefig("Fitted_weibull_T4.png")