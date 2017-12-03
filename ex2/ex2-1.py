import numpy as np
import cv2
import matplotlib.pyplot as plt


def getAffineTransformation(pts1, pts2):
    b = pts2.flatten()
    A = np.zeros((len(b), 6))

    for idx in range(0, len(pts1)):
        A[(idx * 2), 0] = pts1[idx, 0]
        A[(idx * 2), 1] = pts1[idx, 1]
        A[(idx * 2), 4] = 1
        A[(idx * 2) + 1, 2] = pts1[idx, 0]
        A[(idx * 2) + 1, 3] = pts1[idx, 1]
        A[(idx * 2) + 1, 5] = 1

    t_matrix = (np.linalg.lstsq(A, b))[0]

    result = np.array([
        [t_matrix[0], t_matrix[1], t_matrix[4]],
        [t_matrix[2], t_matrix[3], t_matrix[5]],
        [0, 0, 1]])
    return result


def applyAffineTransToImage(img, affineT):
    new_pixel_position = np.zeros((2, img.shape[0] * img.shape[1]))
    inv_affinT = np.linalg.inv(affineT)

    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            res = np.dot(inv_affinT, [c, r, 1])
            new_pixel_position[0, (r * img.shape[0]) + c] = res[0]
            new_pixel_position[1, (r * img.shape[0]) + c] = res[1]

    x = new_pixel_position[0]
    y = new_pixel_position[1]
    x_out_of_range = []
    y_out_of_range = []

    for i in range(0, len(x)):
        if x[i] < 0:
            x[i] = 0
            x_out_of_range.append(i)
        elif x[i] >= img.shape[0] - 1:
            x[i] = img.shape[0] - 2
            x_out_of_range.append(i)

    for i in range(0, len(y)):
        if y[i] < 0:
            y[i] = 0
            y_out_of_range.append(i)
        elif y[i] >= img.shape[1] - 1:
            y[i] = img.shape[1] - 2
            y_out_of_range.append(i)

    x0 = np.int32(x)
    x1 = np.int32(x + 1)
    y0 = np.int32(y)
    y1 = np.int32(y + 1)

    SW = img[(y0, x0)]
    SE = img[(y0, x1)]
    NW = img[(y1, x0)]
    NE = img[(y1, x1)]

    SW[x_out_of_range] = 0
    SE[x_out_of_range] = 0
    NW[x_out_of_range] = 0
    NE[x_out_of_range] = 0
    SW[y_out_of_range] = 0
    SE[y_out_of_range] = 0
    NW[y_out_of_range] = 0
    NE[y_out_of_range] = 0

    u = x - np.float32(x0)
    v = y - np.float32(y0)
    S = SW * (1 - u) + SE * u
    N = NW * (1 - u) + NE * u
    V = S * (1 - v) + N * v

    return np.reshape(V, (img.shape))

# def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, a, b):
#     numOfSegments = Qs.shape[0]
#     Rsrc = np.zeros((img.shape[0] * img.shape[1], 2), np.float32, **None)
#     weightSum = 0
#     for segIdx in np.arange(0, numOfSegments):
#         Q11 = Qs[(segIdx, :)]
#         P11 = Ps[(segIdx, :)]
#         Q12 = Qt[(segIdx, :)]
#         P12 = Pt[(segIdx, :)]
#         (R1, weight1) = deformationBetweenLines(img, Q11, P11, Q12, P12, a, b)
#         Rsrc = Rsrc + R1 * weight1[(:, np.newaxis)]
#         weightSum = weightSum + weight1

#     Rsrc = np.divide(Rsrc, weightSum[(:, np.newaxis)])
#     imgT = interp2Bilinear(img, Rsrc)
#     imgT = np.reshape(imgT, (img.shape[0], img.shape[1]))
#     return imgT


def imGradSobel(img):
    first_col = np.append(np.append(img[0, 0], img[:, 0]), img[-1, - 1])
    last_col = np.append(np.append(img[0, 0], img[:, -1]), img[- 1, - 1])
    first_row = np.append(np.append(img[0, 0], img[0, :]), img[-1, - 1])
    last_row = np.append(np.append(img[0, 0], img[-1, :]), img[- 1, - 1])

    # Reflection Padding
    padded_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    padded_img[0, :] = first_row
    padded_img[-1, :] = last_row
    padded_img[:, 0] = first_col
    padded_img[:, -1] = last_col
    padded_img[1:img.shape[0] + 1, 1:img.shape[0] + 1] = img[:]

    return img


imageName = './images/cameraman.tif'
# img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
# pts1 = np.float32([[0, 5],[20, 30],[15, 12]])
# pts2 = np.float32([[11, 20],[28, 43],[23, 26]])

# affineT = getAffineTransformation(pts1,pts2)
# print (affineT)

# # imgT = applyAffineTransToImage2(img, affineT)
# imgT = applyAffineTransToImage(img, affineT)
# f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')
# ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
# ax2.imshow(imgT, cmap='gray'), ax2.set_title('Transformed')


img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
Gx, Gy, Gmag = imGradSobel(img)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.imshow(img, cmap='gray', vmin=0, vmax=255), ax1.set_title('Original')
ax2.imshow(Gx, cmap='gray', vmin=0, vmax=255), ax2.set_title('Gx')
ax3.imshow(Gy, cmap='gray', vmin=0, vmax=255), ax3.set_title('Gy')
ax4.imshow(Gmag, cmap='gray', vmin=0, vmax=255), ax4.set_title('Gmag')

plt.show()
