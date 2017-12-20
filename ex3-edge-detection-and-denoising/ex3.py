
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, pi
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


####################################################################################################



# to get fixed set of random numbers
np.random.seed(seed=0)


def test_1(imageName):

    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    imageEdges = cv2.Canny(img, 100, 200)

    votesThresh = 60
    distThresh = 15
    radius = range(5, 31, 5)
    circles = HoughCircles(imageEdges, radius, votesThresh, distThresh)

    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255), ax1.set_title(
        'Original+ detected circles')
    ax2.imshow(imageEdges, cmap='gray', vmin=0,
               vmax=255), ax2.set_title('Canny edges')

    circle = []
    for y, x, r, val in circles:
        circle.append(plt.Circle((x, y), r, color=(1, 0, 0), fill=False))
        ax1.add_artist(circle[-1])
    plt.show()


imageName = './Images/coins.tif'
test_1(imageName)
