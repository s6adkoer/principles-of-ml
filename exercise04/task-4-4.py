import numpy as np
import matplotlib.pyplot as plt 

#load blobs
data=np.loadtxt('threeBlobs.csv', dtype=np.float64, comments='#', delimiter=',')

#load faces
matX = np.load('faceMatrix.npy').astype('float')

def FW_UPDATE_Z(X,M,Z, t_max):
    ei= np.identity(Z.shape[0])
    for t in range(t_max):
        G_z = 2* (M.T @ M @ Z - M.T @ X)
        #vectorize for loop
        o=np.argmin(G_z, axis=0)
        Z += 2/(t+2) * (ei[o].T- Z)
    return Z
        
def FW_KMEANS_VERSION1(X,k,T_max):

    idx=np.random.randint(0,X.shape[1],k)
    M=X[:,idx]
    for t in range(T_max):
        Z = 1/k *np.ones((k,X.shape[1]))
        Z = FW_UPDATE_Z(X,M,Z, 1)
        M = X @ Z.T @ np.linalg.inv(Z@Z.T)
    return M,Z
        
## task 4.4.1
#seed for reproducibility
np.random.seed(0)
M,Z = FW_KMEANS_VERSION1(data,3,T_max=10)
colors = [ 'orange','red', 'purple']
#plot different cluster in different colors + their means
for i in range(3):
    plt.scatter(data[0,np.where(Z[i]==1)],data[1,np.where(Z[i]==1)], alpha=0.3,color=colors[i])
    plt.scatter(M[0,i],M[1,i], marker=',', s=80,color=colors[i])
plt.show()

## task 4.4.2
#seed for reproducibility
np.random.seed(0)
M,Z = FW_KMEANS_VERSION1(matX,16,T_max=10)

#plot mean faces 
plot,axs = plt.subplots(4,4, figsize=(4,4))
for i in range(16): 
    pos = axs.ravel()[i].imshow(M[:,i].reshape(19,19), cmap='gray')
    axs.ravel()[i].axis('off')

plt.show()
