# Source Generated with Decompyle++
# File: ex3.pyc (Python 3.6)

'''
Created on Fri Oct 27 10:14:24 2017

@author: 310127444
'''
import numpy as np
from numpy.matlib import repmat
import ex3Utils

def bilateralFilter(imgNoisy, spatial_std, range_std):
    (r, c) = imgNoisy.shape
    imgNoisy = np.float32(imgNoisy)
    kernel_radius = np.int32(np.round(spatial_std * 3))
    (I, J) = np.meshgrid(np.linspace(-kernel_radius, kernel_radius, 2 * kernel_radius + 1), np.linspace(-kernel_radius, kernel_radius, 2 * kernel_radius + 1))
    kernelOffset = np.column_stack((I.flatten(), J.flatten()))
    saptialWeights = np.sqrt(np.sum(kernelOffset ** 2, 1))
    saptialWeights = np.exp(-saptialWeights ** 2 / 2 * spatial_std ** 2)
    saptialWeights = saptialWeights / saptialWeights.sum()
    imgClean = np.zeros((r, c))
    for i in np.arange(0, r):
        for j in np.arange(0, c):
            pixelKernelCoords = np.int32(kernelOffset + repmat([
                i,
                j], kernelOffset.shape[0], 1))
            pixelKernelCoords_i = pixelKernelCoords[(:, 0)]
            pixelKernelCoords_i[pixelKernelCoords_i < 0] = 0
            pixelKernelCoords_i[pixelKernelCoords_i > r - 1] = r - 1
            pixelKernelCoords_j = pixelKernelCoords[(:, 1)]
            pixelKernelCoords_j[pixelKernelCoords_j < 0] = 0
            pixelKernelCoords_j[pixelKernelCoords_j > c - 1] = c - 1
            samples = imgNoisy[(pixelKernelCoords_i, pixelKernelCoords_j)]
            rangeWeights = np.exp(-(samples - imgNoisy[(i, j)]) ** 2 / 2 * range_std ** 2)
            rangeWeights = rangeWeights / rangeWeights.sum()
            finalWeights = saptialWeights * rangeWeights
            imgClean[(i, j)] = np.sum(samples * finalWeights) / np.sum(finalWeights)
        
    
    imgClean = np.uint8(np.round(imgClean))
    return imgClean


def HoughCircles(img, radius, votesThresh, distThresh):
    (M, N) = img.shape
    radius = np.arange(5, 31, 5)
    theta = np.arange(0, 360, 1) * np.pi / 180
    houghDomain = np.zeros((M, N, radius.shape[0]))
    edgePixels = np.transpose(np.nonzero(img))
    numOfEdgePixels = edgePixels.shape[0]
    for edgePxIdx in np.arange(0, numOfEdgePixels):
        y = edgePixels[(edgePxIdx, 0)]
        x = edgePixels[(edgePxIdx, 1)]
        for radIdx in np.arange(0, radius.shape[0]):
            a = y - radius[radIdx] * np.sin(theta)
            b = x - radius[radIdx] * np.cos(theta)
            a = np.uint32(np.round(a))
            b = np.uint32(np.round(b))
            outOfDomainIdx = np.zeros(a.shape)
            outOfDomainIdx[a < 0] = 1
            outOfDomainIdx[a >= M] = 1
            outOfDomainIdx[b < 0] = 1
            outOfDomainIdx[b >= N] = 1
            a = a[outOfDomainIdx == 0]
            b = b[outOfDomainIdx == 0]
            houghDomain[(a, b, radIdx)] += 1
        
    
    circles = np.argwhere(houghDomain > votesThresh)
    vals = houghDomain[(circles[(:, 0)], circles[(:, 1)], circles[(:, 2)])]
    circles[(:, 2)] = radius[circles[(:, 2)]]
    circles = np.vstack((circles.T, vals)).T
    circlesClean = ex3Utils.selectLocalMaxima(circles, distThresh, votesThresh)
    return circlesClean

