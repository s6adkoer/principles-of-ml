def gaussKernelMatrix(u,v,alpha,sigma):
    return alpha*(np.exp(-diffMatrix(u,v)**2/(2*sigma**2)))