import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img
import matplotlib.pyplot as plt
imgF = iio.imread('tree.png', mode='L').astype(float)#np.float raises warning
imgL = iio.imread('lightning.png', mode='L').astype(float)

def binarize(imgF):
    imgD = np.abs(img.filters.gaussian_filter(imgF, sigma=0.50) - \
                  img.filters.gaussian_filter(imgF, sigma=1.00))
    return img.morphology.binary_closing(np.where(imgD < 0.1*imgD.max(), 0, 1))

imgF_binary=binarize(imgF)
imgL_binary=binarize(imgL)

def box_counting(img):
    w, h = img.shape
    n_ls = []
    # l runs from 1 to 9-2 =7
    for l in range(1,8):
        s_l = 1/2**l
        # get box sizes
        box_sizeW = s_l * w
        box_sizeH = s_l * h
        n_l=0
        #each l has 2**l boxes
        for box_w in range(0,(2**l)):
            for box_h in range(0,(2**l)):
                #check if any value in the box is equal 1. If so increment n_l by one
                if (np.any(img[int(box_w * box_sizeW): int((box_w+1) * box_sizeW),\
                               int(box_h * box_sizeH): int((box_h+1) * box_sizeH)]==1)):
                    n_l+=1
        n_ls.append(n_l)
    return n_ls
                    
n_l=box_counting(imgF_binary)
n_l2=box_counting(imgL_binary)
def slope(n_l):
    s_l=[2**l for l in range(1,8)]
    matX=np.vander(np.log(s_l),2,increasing=True)
    b,D=la.lstsq(matX, np.log(n_l),rcond=None)[0]
    return b,D


b,d=slope(n_l)
b2,d2=slope(n_l2)

# Plotting the results

fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

z=[(d*np.log((2**l))+ b) for l in range(1,8)]

ax1.plot([2**l for l in range(1,8)],np.exp(z), '--')
ax1.plot([2**l for l in range(1,8)], (n_l),'s')
ax1.set_yscale("log")
ax1.set_ylabel('log n')
ax1.set_xscale("log")
ax1.set_xlabel(r'log $\frac{1}{s}$')
ax1.set_title(f'tree')
ax1.legend()

z2=[(d2*np.log((2**l))+ b2) for l in range(1,8)]
ax2.plot([2**l for l in range(1,8)],np.exp(z2), '--')

ax2.plot([2**l for l in range(1,8)], (n_l2),'s')
ax2.set_yscale("log")
ax2.set_ylabel('log n')
ax2.set_xscale("log")
ax2.set_xlabel(r'log $\frac{1}{s}$')
ax2.set_title('lightning')
plt.show()

print(f'tree slope :{d}, lightning slope :{d2}')