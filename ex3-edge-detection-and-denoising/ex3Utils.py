# IMPR 2017, IDC
# ex3 utils

import numpy as np
from  scipy.spatial.distance import cdist as cdist

def selectLocalMaxima (circles,votesThresh,distThresh):
    circles = circles[np.argsort(-circles[:, 3]),:]
    
    circlesClean = np.empty((1,4),dtype=np.float32)
    circlesClean[0,:] = circles[0,:]
    
    for cIdx in np.arange(1,circles.shape[0]):
        c = np.empty((1,4),dtype=np.float32)
        c[0,:] = circles[cIdx,:]
        cDist = cdist(c[:,0:3], circlesClean[:,0:3], 'euclidean')
        if cDist.min()>distThresh and c[0,3] > votesThresh :
            circlesClean=np.vstack((circlesClean,c))

    return circlesClean  