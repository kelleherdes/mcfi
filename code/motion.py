import numpy as np
from numba import jit, cuda
import cv2



@cuda.jit
def cuda_block(I1, I2, points, b, max_motion):
    y, x = cuda.grid(2)

    if(y + b >= I1.shape[0] or x + b >= I2.shape[1]):
        return None
    if(y < b or x < b):
        return None
    closest_d  = 10**10 + 0.5
    #max_motion = max_
    mk_width   = 17
    d          = 0
    dx         = 0
    dy         = 0
    for i in range(-max_motion, max_motion):
        for j in range(-max_motion, max_motion):
            for a in range(y - mk_width, y + mk_width + 1):
                for c in range(x - mk_width, x + mk_width + 1):
                    d += (I1[a + i, c + j] - I2[a, c]) ** 2
            if(d < closest_d):
                closest_d = d
                dy        = i
                dx        = j
            d = 0
    points[y, x, 0] = dy
    points[y, x, 1] = dx

@cuda.jit
def bi_cuda_block(I1, I2, points, b, max_motion):
    y, x = cuda.grid(2)

    if(y + b >= I1.shape[0] or x + b >= I2.shape[1]):
        return None
    if(y < b or x < b):
        return None
    closest_d  = 10**10 + 0.5
    #max_motion = max_
    mk_width   = 17
    d          = 0
    dx         = 0
    dy         = 0
    for i in range(-max_motion, max_motion):
        for j in range(-max_motion, max_motion):
            for a in range(y - mk_width, y + mk_width + 1):
                for c in range(x - mk_width, x + mk_width + 1):
                    d += (I1[a + i, c + j] - I2[a - i, c - j]) ** 2
            if(d < closest_d):
                closest_d = d
                dy        = i
                dx        = j
            d = 0
    points[y, x, 0] = dy
    points[y, x, 1] = dx

@cuda.jit
def bi_cuda_block_p(I1, I2, points, b, max_motion, init):
    y, x = cuda.grid(2)
    if(y + b >= I1.shape[0] or x + b >= I2.shape[1]):
        return None
    if(y < b or x < b):
        return None
    closest_d  = 10**10 + 0.5
    #max_motion = max_
    mk_width   = 17
    d          = 0
    dx         = 0
    dy         = 0
    for i in range(-max_motion, max_motion):
        for j in range(-max_motion, max_motion):
            for a in range(y - mk_width, y + mk_width + 1):
                for c in range(x - mk_width, x + mk_width + 1):
                    d += (I1[a + i + init[i, j, 0], c + j + init[i, j, 1]] - I2[a - i - init[i, j, 0], c - j - init[i, j, 1]]) ** 2
            if(d < closest_d):
                closest_d = d
                dy        = i + init[i, j, 0]
                dx        = j + init[i, j, 1]
            d = 0
    points[y, x, 0] = dy
    points[y, x, 1] = dx
    
def motion_est(i1, i2, b, max_):
    TPB = (16, 16)
    bpg_y = int(np.ceil((i1.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((i1.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    assert(i1.shape == i2.shape)
    points = np.zeros((i1.shape[0], i1.shape[1], 2))
    np.ascontiguousarray(points, dtype=np.int16)
    I1 = np.copy(i1).astype(np.uint8)
    I2 = np.copy(i2).astype(np.uint8)
    cuda_block[BPG, TPB](I1, I2, points, b, max_)
    return points.astype(np.int64)

def motion_est2(i1, i2, b, max_):
    TPB = (16, 16)
    bpg_y = int(np.ceil((i1.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((i1.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    assert(i1.shape == i2.shape)
    points = np.zeros((i1.shape[0], i1.shape[1], 2))
    np.ascontiguousarray(points, dtype=np.int16)
    I1 = np.copy(i1).astype(np.uint8)
    I2 = np.copy(i2).astype(np.uint8)
    bi_cuda_block[BPG, TPB](I1, I2, points, b, max_)
    return points.astype(np.int64)


def motion_est_p(i1, i2, b, max_, init):
    TPB = (16, 16)
    bpg_y = int(np.ceil((i1.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((i1.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    assert(i1.shape == i2.shape)
    points = np.zeros((i1.shape[0], i1.shape[1], 2))
    np.ascontiguousarray(points, dtype=np.int16)
    init_c = np.ascontiguousarray(init, dtype=np.int16)
    I1 = np.copy(i1).astype(np.uint8)
    I2 = np.copy(i2).astype(np.uint8)
    bi_cuda_block_p[BPG, TPB](I1, I2, points, b, max_, init_c)
    return points

def motion_pyramid(i1, i2, b, max_):
    #first downsampled image in pyramid
    assert(i1.shape[0] == i2.shape[0])
    assert(i1.shape[1] == i2.shape[1])
    #ensure that frame is divisible by 4
    y_trim = i1.shape[0] % 4
    x_trim = i1.shape[1] % 4

    if(y_trim == 0):
        y_trim = -i1.shape[0]
    if(x_trim == 0):
        x_trim = -i1.shape[1]

    i1_copy = np.copy(i1)[:-y_trim, :-x_trim].astype(np.uint8)
    i2_copy = np.copy(i2)[:-y_trim, :-x_trim].astype(np.uint8)

    i1_1 = cv2.resize(i1_copy, (int(i1.shape[0] / 2), int(i1.shape[1] / 2))).astype(np.uint8)
    i2_1 = cv2.resize(i2_copy, (int(i2.shape[0] / 2), int(i2.shape[1] / 2))).astype(np.uint8)
    #second downsampled image in pyramid
    i1_2 = cv2.resize(i1_1, (int(i1_1.shape[0] / 2), int(i1_1.shape[1] / 2))).astype(np.uint8)
    i2_2 = cv2.resize(i2_1, (int(i2_1.shape[0] / 2), int(i2_1.shape[1] / 2))).astype(np.uint8)

    mvs2 = motion_est2(i1_2, i2_2, b, max_)
    mvs1 = motion_est_p(i1_1, i2_1, b, max_, upsample(mvs2))
    mvs0 = motion_est_p(i1_copy, i2_copy, b, max_, upsample(mvs1))

    padded_mvs = np.zeros((i1.shape[0], i1.shape[1], 2))

    padded_mvs[:-y_trim, :-x_trim] = mvs0
    return mvs0.astype(np.int16)


def upsample(mvs):
    up_mvs = np.zeros((2 * mvs.shape[0], 2 * mvs.shape[1], 2))

    for i in range(0, up_mvs.shape[0]):
        for j in range(0, up_mvs.shape[1]):
            up_mvs[i, j, 0] = 2 * mvs[int(i/2), int(j/2), 0]
            up_mvs[i, j, 1] = 2 * mvs[int(i/2), int(j/2), 1]
    
    return up_mvs




def main():
    i1 = cv2.imread('soccer/image0.png')
    i2 = cv2.imread('soccer/image1.png')
    b = 50
    I1 = cv2.copyMakeBorder(i1[:, :, 1], b, b, b, b, cv2.BORDER_CONSTANT)
    I2 = cv2.copyMakeBorder(i2[:, :, 1], b, b, b, b, cv2.BORDER_CONSTANT)
    print("Estimating motion...")
    points = motion_est(I1, I2, b)
    print("Interpolating")
    output = bilinear(i2, points, b)
    cv2.imwrite('motion.png', output)


