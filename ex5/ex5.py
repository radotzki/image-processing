from scipy import signal
import numpy as np

def getKernel(filterParam):
    return np.array([0.25 - filterParam / 2, 0.25, filterParam, 0.25, 0.25 - filterParam / 2])

def getKernelMatrix(kernel1D):
    k = np.array(kernel1D)[np.newaxis]
    return k.T * k

def imConv2(img, kernel1D):
    kernel = getKernelMatrix(kernel1D)
    return signal.convolve2d(img, kernel, mode='same')

def gaussianPyramid(img, numOfLevels, filterParam):
    G = {}
    G[0] = img
    kernel1D = getKernel(filterParam)

    for i in range(1, numOfLevels):
        G[i] = imConv2(G[i - 1], kernel1D)[::2, ::2]

    return G

def imageUpsampling(img, filterParam):
    padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), np.float32)
    padded[::2, ::2] = img[:, :]
    return 4 * imConv2(padded, getKernel(filterParam))

def laplacianPyramid(img, numOfLevels, filterParam):
    G = gaussianPyramid(img, numOfLevels, filterParam)
    L = {}

    for i in range(0, numOfLevels - 1):
        L[i] = G[i] - imageUpsampling(G[i + 1], filterParam)

    L[numOfLevels - 1] = G[numOfLevels - 1]

    return L

def imgFromLaplacianPyramid(l_pyr, numOfLevels, filterParam):
    img = l_pyr[numOfLevels - 1]
    for i in range(numOfLevels - 2, -1, -1):
        img = l_pyr[i] + imageUpsampling(img, filterParam)

    return img

def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam):
    g_mask = gaussianPyramid(blendingMask, numOfLevels, filterParam)
    l_img1 = laplacianPyramid(img1, numOfLevels, filterParam)
    l_img2 = laplacianPyramid(img2, numOfLevels, filterParam)

    r = {}

    for i in range(0, numOfLevels):
        r[i] = g_mask[i] * l_img1[i] + (1 - g_mask[i]) * l_img2[i]

    return imgFromLaplacianPyramid(r, numOfLevels, filterParam)
