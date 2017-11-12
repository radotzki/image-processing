import numpy as np

def getSampledImageAtResolution(dim, pixel_size, k=2):
    x = np.linspace(dim[0], dim[1], (dim[1] - dim[0]) / pixel_size)
    y = np.linspace(dim[2], dim[3], (dim[3] - dim[2]) / pixel_size)
    X, Y = np.meshgrid(x, y)
    I = np.cos(k * np.pi * (3 * X + 2 * Y))
    return I

def optimalQuantizationImage(img, k):
    z = []
    for i in range(0, 32):
        z.append(np.uint8(i * 256 / k))
    z.append(255)

    q = np.zeros(k)
    q_prev = np.ones(k)
    img_histogram = np.zeros(256)

    for i in range(0, 255):
        img_histogram[i] = np.sum(img == i)

    while np.sum(np.abs(q - q_prev)) > 0.1:
        q_prev = np.copy(q)

        for i in range(0, k):
            zi_hist = img_histogram[z[i]:z[i + 1]]
            if np.sum(zi_hist) > 0:
                q[i] = np.sum(np.arange(z[i], z[i + 1], 1) * zi_hist) / np.sum(zi_hist)
            else:
                q[i] = 0

        for i in range(1, k):
            z[i] = np.uint8((q[i - 1] + q[i]) / 2)

    hash_table = np.empty(256)
    result = np.zeros((np.shape(img)[0],np.shape(img)[1]))

    for i in range(1, k + 1):
        hash_table[z[i - 1]:z[i]] = np.floor(q[i - 1])

    for row in range(0, np.shape(img)[0]):
        for col in range(0, np.shape(img)[1]):
            pixel = img[row, col, 1]
            result[row,col] = np.uint8(hash_table[pixel])

    return result
