import numpy as np
import matplotlib.pyplot as plt

# Helper functions, to shorten the code.

@np.vectorize
def exponential_helper(d,alpha,beta):
    return (d/beta)**alpha

@np.vectorize
def exponential_logarithmic_helper(d,alpha,beta):
    return ((d/beta)**alpha) * np.log(d/beta)

@np.vectorize
def exponential_logarithmic_squared_helper(d,alpha,beta):
    return ((d/beta)**alpha) * (np.log(d/beta)**2)

def get_partials(D,alpha,beta):
    """Calculate partial derivatives as explained on the slides"""
    assert alpha !=0
    assert beta >0
    assert np.min(D)>0

    N=np.sum(D[:,1])

    partial_L_partial_alpha=N/alpha - N*np.log(beta) + np.sum(D[:,1]*np.log(D[:,0]))-np.sum(D[:,1]*exponential_logarithmic_helper(D[:,0],alpha,beta))
    partial_L_partial_beta=alpha/beta*(np.sum(D[:,1]*exponential_helper(D[:,0],alpha,beta))-N)
    partial_partial_l_partial_alpha_alpha=-N/(alpha*alpha)-np.sum(D[:,1]*exponential_logarithmic_squared_helper(D[:,0],alpha,beta))
    partial_partial_l_partial_beta_beta=alpha/(beta*beta)*(N-(alpha+1)*np.sum(D[:,1]*exponential_helper(D[:,0],alpha,beta)))
    partial_partial_L_partial_alpha_beta=np.sum(D[:,1]*exponential_helper(D[:,0],alpha,beta))/beta + alpha/beta * np.sum(D[:,1]*exponential_logarithmic_helper(D[:,0],alpha,beta))-N/beta

    return np.array([partial_L_partial_alpha,partial_L_partial_beta]),np.array([[partial_partial_l_partial_alpha_alpha,partial_partial_L_partial_alpha_beta],[partial_partial_L_partial_alpha_beta,partial_partial_l_partial_beta_beta]])

def update(D,theta_hat,steps=20):
    for _ in range(steps):
        partials,M=get_partials(D,*theta_hat)
        theta_hat=theta_hat-np.linalg.inv(M.T)@(partials)

    return theta_hat

@np.vectorize
def f(t,alpha,beta):
    """Calculate the density at point t for a given weibull distribution"""
    return (alpha/beta)*np.power((t/beta),(alpha-1))*np.exp(-np.power((t/beta),alpha))

data = data=np.loadtxt("myspace.csv",dtype=object,delimiter=",")

time_frame = data[:,0].astype(str)
global_interest = data[:,1].astype(float)

to_skip=np.where(global_interest != 0) # There are no zeros after the leading zeros, therefore this is fine


ts=np.array([1+i for i in range(np.array(to_skip).shape[1])])

h_of_t=np.stack((ts,global_interest[to_skip]),axis=1)

theta_hat=update(h_of_t,np.array([1,1]),steps=19)

print(theta_hat)

f_tilde= lambda t: f(t,*theta_hat)

plt.clf()

plt.plot(ts,global_interest[to_skip],label="Myspace data",c="gray")
plt.plot(ts,f_tilde(ts)*np.sum(h_of_t[:,1]),label="Fitted Weibull",c="red")
plt.xlabel("Timestep")
plt.ylabel("Global Interest")
plt.legend()
plt.gca().set_aspect(1)
plt.savefig("Fitted_weibull_T2.png")