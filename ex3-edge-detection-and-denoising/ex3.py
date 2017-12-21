import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, pi
from matplotlib.backends.backend_pdf import PdfPages
import ex3Utils

def HoughCircles(imageEdges, radius, votesThresh, distThresh):
    A = np.zeros((imageEdges.shape[0], imageEdges.shape[1], len(radius)))
    edges_indice = np.argwhere(imageEdges > 0)
    T = np.arange(0, 360, 1) * pi / 180
    radius = np.arange(radius.start, radius.stop, radius.step)

    for edge in edges_indice:
        x = edge[1]
        y = edge[0]
        for r_idx in range(0, len(radius)):
            r = radius[r_idx]
            a = y - r * np.sin(T)
            b = x - r * np.cos(T)
            a = np.uint32(np.round(a))
            b = np.uint32(np.round(b))

            out_of_bound_idx = np.zeros(a.shape)
            out_of_bound_idx[a < 0] = 1
            out_of_bound_idx[a > imageEdges.shape[0] - 1] = 1
            out_of_bound_idx[b < 0] = 1
            out_of_bound_idx[b > imageEdges.shape[1] - 1] = 1
            a = a[out_of_bound_idx == 0]
            b = b[out_of_bound_idx == 0]

            A[(a, b, r_idx)] += 1

    circles = np.argwhere(A > votesThresh)
    vals = A[(circles[:, 0], circles[:, 1], circles[:, 2])]
    circles[:, 2] = radius[circles[:, 2]]
    circles = np.vstack((circles.T, vals)).T
    circlesClean = ex3Utils.selectLocalMaxima(circles, distThresh, votesThresh)

    return circlesClean


def bilateralFilter(imgNoisy, spatial_std, range_std):
    filtered_img = np.zeros(imgNoisy.shape)
    radius = 3 * spatial_std

    for i in range(0, imgNoisy.shape[0]):
        for j in range(0, imgNoisy.shape[1]):
            norm = 0
            sigma_iw = 0

            for k in range(max(i - radius, 0), min(i + radius, imgNoisy.shape[0])):
                for l in range(max(j - radius, 0), min(j + radius, imgNoisy.shape[1])):
                    w = weight(imgNoisy, i, j, k, l, spatial_std, range_std)
                    norm += w
                    sigma_iw += imgNoisy[k, l] * w

            filtered_img[i, j] = sigma_iw / norm

    return filtered_img


def weight(I, i, j, k, l, spatial_std, range_std):
    return np.exp(
        - (((i - k)**2 + (j - l)**2) / (2 * spatial_std**2))
        - (((I[i, j] - I[k, l])**2) / (2 * range_std**2))
    )
