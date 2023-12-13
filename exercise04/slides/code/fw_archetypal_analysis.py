def FW_ARCHETYPAL_ANALYSIS(X,k, T_max=100):
    
    idx=np.random.randint(0,X.shape[1],k)
    A=X[:,idx]
    for t in range(T_max):
        Z = 1/k *np.ones((k,X.shape[1]))
        Z = FW_UPDATE_Z(X,A,Z, t_max=100)
        
        Y = 1/X.shape[1] *np.ones((X.shape[1],k))
        Y = FW_update_Y(X, Y,Z, t_max=100)
        
        A = X@Y
    return A,Y,Z