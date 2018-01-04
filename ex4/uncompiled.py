# Source Generated with Decompyle++
# File: ex4.pyc (Python 3.6)

import numpy as np

def Fourier1D(x_n):
    N = x_n.shape[0]
    F = np.zeros(N, np.complex, **None)
    for k in np.arange(0, N):
        for n in np.arange(0, N):
            F[k] = F[k] + x_n[n] * np.complex(np.cos(2 * np.pi * k * n / N), -np.sin(2 * np.pi * k * n / N))
        
    
    return F


def Fourier1DPolar(x_n):
    F = Fourier1D(x_n)
    R = np.sqrt(np.real(F) ** 2 + np.imag(F) ** 2)
    theta = np.arctan2(np.imag(F), np.real(F))
    F_n = np.column_stack((R, theta))
    return F_n


def invFourier1DPolar(F_n_polar):
    a = F_n_polar[(:, 0)] * np.cos(F_n_polar[(:, 1)])
    b = F_n_polar[(:, 0)] * np.sin(F_n_polar[(:, 1)])
    F_n = a + (0+1j) * b
    return invFourier1D(F_n)


def invFourier1D(F_n):
    N = F_n.shape[0]
    x = np.zeros(N, np.complex, **None)
    for k in np.arange(0, N):
        for n in np.arange(0, N):
            x[k] = x[k] + F_n[n] * np.complex(np.cos(2 * np.pi * k * n / N), np.sin(2 * np.pi * k * n / N))
        
    
    x = x / N
    return x


def imageUpsampling(img, upsamplingFactor):
    Fimg = np.fft.fftshift(np.fft.fft2(img))
    (r, c) = Fimg.shape
    if upsamplingFactor[0] >= 1 and upsamplingFactor[1] >= 1:
        zeroPaddedFimg = np.zeros((r * upsamplingFactor[0], c * upsamplingFactor[1]), np.complex, **None)
        center = np.floor(zeroPaddedFimg.shape) / 2
        zeroPaddedFimg[(np.int(center[0] - r / 2):np.int(center[0] + r / 2), np.int(center[1] - c / 2):np.int(center[1] + c / 2))] = Fimg
        zeroPaddedFimg = zeroPaddedFimg * upsamplingFactor[0] * upsamplingFactor[1]
        upsampledImage = np.abs(np.fft.ifft2(zeroPaddedFimg))
    else:
        zeroPaddedFimg = Fimg
        upsampledImage = img
    return (upsampledImage, Fimg, zeroPaddedFimg)


def phaseCorr(img1, img2):
    Fimg1 = np.fft.fft2(img1)
    Fimg2 = np.fft.fft2(img2)
    Fimg2conj = np.conj(Fimg2)
    FphaseCorr = Fimg1 * Fimg2conj / np.abs(Fimg1 * Fimg2conj)
    phaseCorr = np.fft.ifft2(FphaseCorr)
    ind = np.argmax(phaseCorr)
    res_dy = np.int32(ind / phaseCorr.shape[1])
    res_dx = np.mod(ind, phaseCorr.shape[1])
    return (res_dx, res_dy, phaseCorr)


def imFreqFilter(img, lowThresh, highThresh):
    Fimg = np.fft.fftshift(np.fft.fft2(img))
    (u, v) = Fimg.shape
    mask = np.ones((u, v), np.float32, **None)
    (maskX, maskY) = np.meshgrid(np.linspace(0, u - 1, u) - u / 2, np.linspace(0, v - 1, v) - v / 2)
    maskDist = np.sqrt(maskX ** 2 + maskY ** 2)
    mask[maskDist < lowThresh] = 0
    mask[maskDist > highThresh] = 0
    filtImage = np.abs(np.fft.ifft2(Fimg * mask))
    return (filtImage, Fimg, mask)


def imageDeconv(img, h, k):
    H = np.fft.fft2(h, img.shape)
    Fimg = np.fft.fft2(img)
    HConj = np.conj(H)
    (X, Y) = np.meshgrid(np.linspace(0, img.shape[0] - 1, img.shape[0]), np.linspace(1, img.shape[1] - 1, img.shape[1]))
    coordsWeight = X ** 2 + Y ** 2
    Frec = (HConj / (H * HConj + k * coordsWeight)) * Fimg
    rec = np.real(np.fft.ifft2(Frec))
    rec = np.roll(rec, np.int(h.shape[0] / 2), 0)
    rec = np.roll(rec, np.int(h.shape[1] / 2), 1)
    return rec

