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