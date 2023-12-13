def FW_KMEANS_VERSION1(X,k,T_max):

    idx=np.random.randint(0,X.shape[1],k)
    M=X[:,idx]
    for t in range(T_max):
        Z = 1/k *np.ones((k,X.shape[1]))
        Z = FW_UPDATE_Z(X,M,Z, 1)
        M = X @ Z.T @ np.linalg.inv(Z@Z.T)
    return M,Z
        