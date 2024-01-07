import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

import task5plot

#load data
matX=np.loadtxt('threeBlobs.csv', dtype=np.float64, comments='#', delimiter=',')

#frank-wolfe algorithm for minimum enclosing ball
def fwDualMEB(matX, vecZ, T=100):
    m, n = matX.shape 
    vecM = np.ones(n)/n 
    for t in range(T):
        beta = 2 / (t+2) 
        vecG = 2 * matX.T @ matX @ vecM - vecZ 
        imin = np.argmin(vecG) 
        vecM *= (1-beta)
        vecM[imin] += beta
    return vecM

z=np.diag(matX.T @matX)

mu_hat = fwDualMEB(matX, z)

#calculate center and radius
c = matX @ mu_hat
r = np.sqrt(mu_hat.T @ z - mu_hat.T@ matX.T@matX@ mu_hat)


def compChiFnct(matX, vecM, c,r, bbox, nx=512):
    w = bbox['xmax'] - bbox['xmin'] 
    h = bbox['ymax'] - bbox['ymin'] 
    ny = int(nx*h/w) 
    xs, ys = np.meshgrid(np.linspace(bbox['xmin'], bbox['xmax'], nx),
                         np.linspace(bbox['ymin'], bbox['ymax'], ny))
    
    matXtst = np.vstack((xs.flatten(), ys.flatten()))

    #calculate if test point lies outside, inside or on the boundary of the ball 
    vecYtst = la.norm(matXtst-c[...,None],axis=0)**2 - r**2

    return xs, ys, vecYtst.reshape(ny, nx)

bbox=task5plot.compBBox(matX, eps=0.2)
#compute chi_B
chiFnct=compChiFnct(matX, mu_hat, c,r, bbox )
#visualize result
task5plot.plot2dDataFnct([matX], bbox, fctF=chiFnct, showCont=True,
               cmap=cm.coolwarm, cmapalph=0.75)
