import numpy as np
from numba import jit, cuda

@cuda.jit
def cuda_c(I, Q, ac_off, mvs1, b, c):
    y, x = cuda.grid(2)
    r1 = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    for k in range(0, Q.shape[0]):
        for i in range(-ac_off, ac_off + 1):
            for j in range(-ac_off, ac_off + 1):
                r1 += I[y + i + Q[k, 0] - mvs1[y, x, 0], x + j + Q[k, 1] - mvs1[y, x, 1], 0] * I[y + i, x + j, 1] 
        c[y - b, x - b, k]  = r1  
        r1 = 0

@cuda.jit
def cuda_C(I, Q, ac_off, mvs1, b, C):
    y, x = cuda.grid(2)
    r = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    
    for k in range(0, Q.shape[0]):
        for l in range(k, Q.shape[0]):
            for i in range(-ac_off, ac_off + 1):
                for j in range(-ac_off, ac_off + 1):
                    r += I[y + i + Q[l, 0] - mvs1[y, x, 0], x + j + Q[l, 1] - mvs1[y, x, 1], 0] * \
                         I[y + i + Q[k, 0] - mvs1[y, x, 0], x + j + Q[k, 1] - mvs1[y, x, 1], 0] 
            C[y - b, x - b, k, l] = C[y - b, x - b, l, k] = r
            r = 0


def get_c(I, Q, ac_off, mvs1, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0])/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1])/TPB[1]))
    BPG = (bpg_y, bpg_x)
    c = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, Q.shape[0]), dtype = np.int32)
    cuda_c[BPG, TPB](I, Q, ac_off, mvs1, b, c)
    return c

def get_C(I, Q, ac_off, mvs1, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    C = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, Q.shape[0], Q.shape[0]), dtype = np.int32)
    cuda_C[BPG, TPB](I, Q, ac_off, mvs1, b, C)
    return C
