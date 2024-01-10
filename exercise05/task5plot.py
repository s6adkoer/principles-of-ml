
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm



def compBBox(matX, eps=0.1):
    xmin, xmax = np.min(matX[0]), np.max(matX[0])
    ymin, ymax = np.min(matX[1]), np.max(matX[1])
    dltx, dlty = (xmax - xmin) * eps, (ymax - ymin) * eps
    
    return {'xmin':xmin-dltx, 'xmax':xmax+dltx,
            'ymin':ymin-dlty, 'ymax':ymax+dlty}



def plot2dDataFnct(matXlist,  
                   bboxdict,
                   matS=None,
                   fctF=(None,None,None),
                   cmap=clr.ListedColormap(['C0','C1'], 'indexed'),
                   cmapalph=0.25,
                   showAxes=False,
                   showCont=False,
                   showFnct=True,
                   title=None,
                   filename=None):
    fig = plt.figure()
    axs = fig.add_subplot(111, aspect='equal', facecolor='w')

    setAxes(axs) if showAxes else axs.set_axis_off()
    setLims(axs, bboxdict)

    cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    xstl = {'ls':'', 'marker':'o',
            'ms': 7, 'mew':0, 'mec':'k', 'alpha':0.50}
    sstl = {'ls':'', 'marker':'s', 'fillstyle':'none',
            'ms':11, "mew":1, 'mec':'k', 'alpha':1.00}
    xs, ys, fs = fctF

    if fs is not None and showFnct:
        axs.imshow(fs, interpolation='bicubic', origin='lower',
                   extent=(bboxdict['xmin'], bboxdict['xmax'],
                           bboxdict['ymin'], bboxdict['ymax']),
                   cmap=cmap, norm=clr.TwoSlopeNorm(0), alpha=cmapalph)

    # plot data points in data matrices 
    for i, matX in enumerate(matXlist):
        xstl['c'] = 'k' if len(matXlist) <= 1 else cols[i%10]
        axs.plot(matX[0], matX[1], **xstl)


    # plot support vectors in matrix S
    if matS is not None:
        axs.plot(matS[0,:], matS[1,:], **sstl)

    # plot zero contour line
    if xs is not None and \
       ys is not None and \
       fs is not None and \
       showCont:
        axs.contour(xs, ys, fs, [0], colors='k', alpha=1.00)   

    plt.title(title)
    plt.show() if filename is None else writeFigure(fig, filename)
    plt.close()


def setLims(axs, bboxdict):
    axs.set_xlim(bboxdict['xmin'], bboxdict['xmax'])
    axs.set_ylim(bboxdict['ymin'], bboxdict['ymax'])


def setAxes(axs):
    for a in ['right', 'top']:
        axs.spines[a].set_visible(False)
    for a in ['left', 'bottom']:
        axs.spines[a].set_alpha(1)
        axs.spines[a].set_zorder(0)
        axs.spines[a].set_color('k')
        axs.spines[a].set_linewidth(1)
        axs.spines[a].set_position('zero')
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')



def writeFigure(fig, fname, pad=0.1):
    fmt = fname.split('.')[-1]
    fig.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='w',
                format=fmt, transparent=False, bbox_inches='tight', pad_inches=pad)




    
if __name__ == '__main__':
    pass






