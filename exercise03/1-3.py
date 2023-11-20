# %% Imports
import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)

# %% Exercise 3.1.1
def diffMatrix(u, v):
    return u[:, np.newaxis] - v
    
# %% Exercise 3.1.2
def productMatrix(u, v):
    return u[:, np.newaxis] * v

# %% Exercise 3.2.1
def linearKernalMatrix(u, v, alpha):
    return alpha * productMatrix(u, v)

# %% Exercise 3.2.2
def gaussKernelMatrix(u, v, alpha, sigma):
    return alpha * np.exp(-np.square(diffMatrix(u, v)) / (2 * sigma**2))

# %% Exercise 3.3
vecX = np.linspace(-5.0, 15.0, 55)
vec0 = np.zeros_like(vecX)

# %% Exercise 3.3.1
fig, axs = plt.subplots(3, 1)
for i, alpha in enumerate([0.1, 1.0, 10.0]):
    axs[i].set_title(f"Linear kernel with alpha = {alpha}")
    y = np.random.multivariate_normal(vec0, linearKernalMatrix(vecX, vecX, alpha), 5)
    for vecY in y:
        axs[i].scatter(vecX, vecY)
        axs[i].plot(vecX, vecY)
plt.tight_layout()

# %% Exercise 3.3.2
fig, axs = plt.subplots(3, 3)
for (i, alpha), (j, sigma) in it.product(enumerate([0.1, 1.0, 10.0]), enumerate([0.1, 1.0, 10.0])):
    axs[i, j].set_title(f"{alpha=},{sigma=}")
    y = np.random.multivariate_normal(vec0, gaussKernelMatrix(vecX, vecX, alpha, sigma), 5)
    for vecY in y:
        axs[i, j].scatter(vecX, vecY)
        axs[i, j].plot(vecX, vecY)
plt.tight_layout()

# %% Exercise 3.3.3
fig, axs = plt.subplots(9, 3)
for (i, alphaG), (j, sigma), (k, alphaL) in it.product(enumerate([0.1, 1.0, 10.0]), enumerate([0.1, 1.0, 10.0]), enumerate([0.1, 1.0, 10.0])):
    axs[i*3+j, k].set_title(f"{alphaG=},{sigma=},{alphaL=}")
    y = np.random.multivariate_normal(vec0, linearKernalMatrix(vecX, vecX, alphaL)+gaussKernelMatrix(vecX, vecX, alphaG, sigma), 5)
    for vecY in y:
        axs[i*3+j, k].scatter(vecX, vecY)
        axs[i*3+j, k].plot(vecX, vecY)
plt.tight_layout()

# %%
