import numpy as np
import numpy.matlib
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
    V = interBiLiner(img, x, y)

    return np.reshape(V, (img.shape))

def interBiLiner(img, x, y):
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
    return V

def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b):
    R_sum = [0,0]
    W_sum = [0,0]

    for point_idx in range(0, len(Qs)):
        Q = Qs[point_idx]
        P = Ps[point_idx]
        Q_ = Qt[point_idx]
        P_ = Pt[point_idx]

        u = (Q - P) / np.linalg.norm(Q - P)
        v = np.asarray([u[1], -u[0]])
        u_ = (Q_ - P_) / np.linalg.norm(Q_ - P_)
        v_ = np.asarray([u_[1], -u_[0]])

        BETA = []
        R = []

        for x in range(0, 256):
            for y in range(0, 256):
                R_ = np.array([x, y])

                alpha = np.dot(R_ - np.matlib.repmat(P_, R_.shape[0], 1), u_) / np.linalg.norm(Q_ - P_)
                beta = np.dot(R_ - np.matlib.repmat(P_, R_.shape[0], 1), v_)
                BETA.append(beta)
                R.append(P + np.multiply(np.multiply(alpha, np.linalg.norm(Q - P)), u) + np.multiply(beta, v))

        R = np.array(R)
        BETA = np.array(BETA)

        Wi = ((np.linalg.norm(Q - P) ** p) / (0.001 + BETA)) ** b
        R_sum += R * Wi
        W_sum += Wi

    R = R_sum / W_sum
    imgT = interBiLiner(img, R[:, 0], R[:, 1])
    imgT = np.reshape(imgT, (img.shape[0], img.shape[1]), order='F')
    return imgT

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

    # Sobel
    grad_x = padded_img[:, 2:] - padded_img[:, :-2]
    grad_x = grad_x[:-2, :] + grad_x[2:, :] + 2 * grad_x[1:-1, :]
    grad_y = padded_img[2:, :] - padded_img[:-2, :]
    grad_y = grad_y[:, :-2] + grad_y[:, 2:] + 2 * grad_y[:, 1:-1]
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return (grad_x, grad_y, grad_magnitude)
