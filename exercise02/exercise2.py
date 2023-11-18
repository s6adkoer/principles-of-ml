import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = data=np.loadtxt("myspace.csv",dtype=object,delimiter=",")

time_frame = data[:,0].astype(str)
global_interest = data[:,1].astype(float)

to_skip=np.where(global_interest != 0)
ts=np.array([1+i for i in range(np.array(to_skip).shape[1])])
hs=global_interest[to_skip]

@np.vectorize
def f(t,alpha,beta):
    return (alpha/beta)*np.power((t/beta),(alpha-1))*np.exp(-np.power((t/beta),alpha))

theta_2=(2.809, 215.429)
theta_3=(2.731, 211.024)
theta_4=(2.646, 213.454)
theta_5=(2.87, 215.91)


tasks=["2.2","2.3","2.4","2.5"]

fs=[lambda x: f(x,*theta_2),lambda x: f(x,*theta_3),lambda x: f(x,*theta_4),lambda x: f(x,*theta_5)]

for i in range(4):
    plt.plot(ts,fs[i](ts),label=f"Fitted Weibull: Task {tasks[i]}")
plt.xlabel("Timestep")
plt.ylabel("Global Interest")
plt.legend()
plt.gca().set_aspect(100/max(fs[0](ts)))
plt.show()