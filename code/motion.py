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
    mk_width   = 10
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

def cuda_block2(I1, I2, points, b, max_motion):
    y, x = cuda.grid(2)

    if(y + b >= I1.shape[0] or x + b >= I2.shape[1]):
        return None
    if(y < b or x < b):
        return None
    closest_d  = 10**10 + 0.5
    #max_motion = max_
    mk_width   = 10
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
    
def motion_est(i1, i2, b, max_):
    TPB = (16, 16)
    bpg_y = int(np.ceil((i1.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((i1.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    assert(i1.shape == i2.shape)
    points = np.zeros((i1.shape[0], i1.shape[1], 2))
    np.ascontiguousarray(points, dtype=np.int32)
    I1 = np.copy(i1)
    I2 = np.copy(i2)
    #np.ascontiguousarray(I1, dtype=np.int32)
    #np.ascontiguousarray(I2, dtype=np.int32)
    cuda_block[BPG, TPB](I1, I2, points, b, max_)
    return points.astype(np.int64)

def motion_est2(i1, i2, b, max_):
    TPB = (16, 16)
    bpg_y = int(np.ceil((i1.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((i1.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    assert(i1.shape == i2.shape)
    points = np.zeros((i1.shape[0], i1.shape[1], 2))
    np.ascontiguousarray(points, dtype=np.int32)
    I1 = np.copy(i1)
    I2 = np.copy(i2)
    #np.ascontiguousarray(I1, dtype=np.int32)
    #np.ascontiguousarray(I2, dtype=np.int32)
    cuda_block2[BPG, TPB](I1, I2, points, b, max_)
    return points.astype(np.int64)

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


