def fwDualMEB2(K,k, T=100):
    m, n = K.shape 
    vecM = np.ones(n)/n 
    for t in range(T):
        beta = 2 / (t+2) 
        vecG = K @ vecM - k
        imin = np.argmin(vecG) 
        vecM *= (1-beta)
        vecM[imin] += beta
    return vecM