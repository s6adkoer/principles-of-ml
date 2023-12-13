def FW_UPDATE_Z(X,M,Z, t_max):
    ei= np.identity(Z.shape[0])
    for t in range(t_max):
        G_z = 2* (M.T @ M @ Z - M.T @ X)
        #vectorize for loop
        o=np.argmin(G_z, axis=0)
        Z += 2/(t+2) * (ei[o].T- Z)
    return Z