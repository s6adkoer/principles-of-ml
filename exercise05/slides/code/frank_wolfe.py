def fwDualMEB(matX,vecZ,T=100):
    m, n = matX.shape 
    vecM = np.ones(n)/n 
    for t in range(T):
        beta = 2 / (t+2) 
        vecG = 2 * matX.T @ \
            matX @ vecM - vecZ 
        imin = np.argmin(vecG) 
        vecM *= (1-beta)
        vecM[imin] += beta
    return vecM
