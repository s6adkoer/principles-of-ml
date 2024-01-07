import scipy.spatial as spt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

import task5plot

matX=np.loadtxt('threeBlobs.csv', dtype=np.float64, comments='#', delimiter=',')


def SQEDMAB(A,B):
    """ if matrices A and B are of shapes (m, na) and (m, nb), then this 
    functions returns a squared distance matrix of shape (na, nb) """ 
    return spt.distance.cdist(A.T, B.T, 'sqeuclidean')

def gaussKernelVector(matXtst, matXtrn, **kPars):
    sigm = kPars['sigm'] if 'sigm' in kPars else 1 
    if matXtst.ndim == 1:
        dist = np.sum((matXtrn.T-matXtst)**2, axis=1) 
    else:
        dist = SQEDMAB(matXtst, matXtrn)
    return np.exp(-0.5/sigm**2 * dist)

    
def gaussianKernelMatrix(matXtrn, sigma):
    return np.exp(- 1/(2*sigma**2)* la.norm(matXtrn.T[...,None]- matXtrn[None,...], axis=1)**2)

sigma=1
kParams = {'sigm' : sigma} 

K=gaussianKernelMatrix(matX,sigma)

#np.diag(gaussKernelVector(matX, matX, **kParams))= np.ones(matX.shape[1])
k=np.ones(matX.shape[1])


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

mu=fwDualMEB2(K,k, T=100)


def compChiFnct2(matX, vecM,K,k,gK,gV,sigma, bbox, nx=512):
    w = bbox['xmax'] - bbox['xmin'] 
    h = bbox['ymax'] - bbox['ymin'] 
    ny = int(nx*h/w) 
    
    xs, ys = np.meshgrid(np.linspace(bbox['xmin'], bbox['xmax'], nx),
                         np.linspace(bbox['ymin'], bbox['ymax'], ny))

    
    matXtst = np.vstack((xs.flatten(), ys.flatten()))

    #kernelized response function
    #np.diag(gK(matXtst,sigma)) == np.ones(matXtst.shape[1])
    
    vecYtst = np.ones(matXtst.shape[1]) - (2* gV(matXtst,matX,**{'sigm' : sigma} ) @ vecM) - vecM.T @ k + 2* vecM.T @ K @ vecM
    
    return xs, ys, vecYtst.reshape(ny, nx)

gK=gaussianKernelMatrix
gV = gaussKernelVector

bbox=task5plot.compBBox(matX, eps=0.3)
#compute chi_B
chiFnct=compChiFnct2(matX, mu, K,k,gK,gV,sigma, bbox )
#visualize result
task5plot.plot2dDataFnct([matX], bbox, fctF=chiFnct, showCont=True,
               cmap=cm.coolwarm, cmapalph=0.75)
