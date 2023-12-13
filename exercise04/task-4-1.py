import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, lstsq

plt.rcParams.update({
    "text.usetex": True
})

X=np.loadtxt("threeBlobs.csv",dtype=np.float,delimiter=",")

mu=np.mean(X,axis=1)

def grad_g(w,n):
    vecW=np.ones(n)/n
    return 2*X.T@X@(w-vecW)

def frank_wolfe_minimize_avg_dist(X,T=10000):
    n=X[0,:].shape[0]
    wt=np.zeros(n)
    wt[0]=1
    for t in range(T):
        gamma_t=2/(t+2)
        imin=np.argmin(grad_g(wt,n))
        wt *= gamma_t
        wt[imin] += gamma_t
    return X@wt
    

TS=[10, 100, 1000, 10000]
results=[]
l2_errors=[]

for T in TS: # Calculate estimated minimizers and errors

    res=frank_wolfe_minimize_avg_dist(X,T=T)
    l2_error=norm(res-mu)

    results.append(res)
    l2_errors.append(l2_error)

    print(f"T={T}")
    print(f"result = {res}, l2 error = {l2_error}")


    plt.clf()
    plt.scatter(X[0,:],X[1,:],label="data",alpha=0.5,c="black")
    plt.scatter(res[0],res[1],label="Frank-Wolfe-result",alpha=0.7,marker="s")
    plt.scatter(mu[0],mu[1],label="Mean",alpha=0.5,marker="v",c="black")

    plt.legend()
    plt.savefig(f"./images/{T}.png")

# Plot errors

plt.clf()
plt.loglog(TS,l2_errors,"--",label="$l^2$ errors",c="black")
plt.loglog(TS,[1/T for T in TS],"--",label="$ \\frac{1}{T} $",c="gray")
plt.xlabel("T")
plt.ylabel("Error")
plt.legend()
plt.savefig("./images/errors.png")

# estimate exponent of error function similarly to Task 1-3

l2_errors=np.array(l2_errors)
TS=np.array(TS)

M=np.stack([np.ones_like(l2_errors),np.log(1/l2_errors)],axis=1)
x,*_=lstsq(M,np.log(TS))

print(f"Least squares: {x[1]} ~ 1")