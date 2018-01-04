
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib.backends.backend_pdf import PdfPages


def Fourier1D(x_n):
    N = len(x_n)
    F_n = np.zeros(N, np.complex)

    for k in range(0, N):
        Xk = 0

        for n in range(0, N):
            Xk += x_n[n] * np.complex(np.cos((2 * np.pi * k * n) / N), -
                                      np.sin((2 * np.pi * k * n) / N))

        F_n[k] = Xk

    return F_n


def invFourier1D(F_n):
    N = len(F_n)
    x_n_rev = np.zeros(N, np.complex)

    for k in range(0, N):
        xn = 0

        for n in range(0, N):
            xn += np.complex(F_n[n] * np.cos((2 * np.pi * k * n) / N),
                             F_n[n] * np.sin((2 * np.pi * k * n) / N)) / N

        x_n_rev[k] = xn

    return x_n_rev


def Fourier1DPolar(x_n):
    c = Fourier1D(x_n)
    F_n = np.zeros((len(c), 2))

    for i in range(0, len(c)):
        r = np.sqrt(c[i].real ** 2 + c[i].imag ** 2)
        theta = np.arctan2(c[i].imag, c[i].real)
        F_n[i] = np.array([r, theta])

    return F_n


def invFourier1DPolar(F_n_polar):
    F_n = np.zeros(len(F_n_polar), np.complex)

    for i in range(0,  len(F_n_polar)):
        x = F_n_polar[i, 0] * np.cos(F_n_polar[i, 1])
        y = F_n_polar[i, 0] * np.sin(F_n_polar[i, 1])
        F_n[i] = np.complex(x, y)

    return invFourier1D(F_n)


def imageUpsampling(img, upsamplingFactor):
    f_img = np.fft.fftshift(np.fft.fft2(img))
    if upsamplingFactor[0] >= 1 and upsamplingFactor[1] >= 1:
        padded_f_img = np.zeros(
            (img.shape[0] * upsamplingFactor[0], img.shape[1] * upsamplingFactor[1]), np.complex)
        x = int((padded_f_img.shape[0] - f_img.shape[0]) / 2)
        y = int((padded_f_img.shape[1] - f_img.shape[1]) / 2)
        padded_f_img[x:x + f_img.shape[0], y:y + f_img.shape[1]] = f_img
        padded_f_img = padded_f_img * upsamplingFactor[0] * upsamplingFactor[1]
        upsampled_img = np.abs((np.fft.ifft2(padded_f_img)))
    else:
        padded_f_img = f_img
        upsampled_img = img
    return (upsampled_img, f_img, padded_f_img)


def phaseCorr(img1, img2):
    ga = np.fft.fft2(img1)
    gb = np.fft.fft2(img2)
    gb_ = np.conjugate(gb)
    R = np.multiply(ga, gb_) / np.abs(np.multiply(ga, gb_))
    r = np.fft.ifft2(R)
    argmax = np.argmax(r)
    return argmax % img1.shape[1], int(argmax / img1.shape[1]), r


def imFreqFilter(img, lowThresh, highThresh):
    mask = np.ones(img.shape)

    for u in range(int(-mask.shape[0] / 2), int(mask.shape[0] / 2)):
        for v in range(int(-mask.shape[1] / 2), int(mask.shape[1] / 2)):
            D_uv = np.sqrt(u ** 2 + v ** 2)
            if D_uv < lowThresh or D_uv > highThresh:
                mask[u + int(mask.shape[0] / 2), v +
                     int(mask.shape[1] / 2)] = 0

    fimg = np.fft.fftshift(np.fft.fft2(img))
    res = np.abs(np.fft.ifft2(fimg * mask))

    return res, fimg, mask


def print_pdf():
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    thresh = [
        [0, 30],
        [0, 40],
        [0, 50],
        [10, 400],
        [20, 400],
        [20, 300],
        [20, 40],
        [40, 60],
        [60, 80],
    ]
    imgs = []

    for i in range(0, len(thresh)):
        filtImage, Fimg, mask = imFreqFilter(img, thresh[i][0], thresh[i][1])
        imgs.append(filtImage)

    f, ax = plt.subplots(3, 3, sharex='col', sharey='row')

    for r in range(0, ax.shape[0]):
        for c in range(0, ax.shape[1]):
                ax[r,c].imshow(imgs[ax.shape[1] * r + c], cmap='gray', vmin=0, vmax=255), ax[r,c].set_title(thresh[ax.shape[1] * r + c])

    plt.figure()
    plt.clf()

    pp = PdfPages('filtered_images.pdf')
    pp.savefig(f)

################################################################################
