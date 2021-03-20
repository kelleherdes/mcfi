import numpy as np
from numba import jit, cuda

@cuda.jit
def cuda_c_m(I, Q, ac_off, mvs1, mvs2, b, c):
    y, x = cuda.grid(2)
    r1 = 0
    r2 = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    for k in range(0, Q.shape[0]):
        for i in range(-ac_off, ac_off + 1):
            for j in range(-ac_off, ac_off + 1):
                r1 += I[y + i + Q[k, 0] + mvs1[y, x, 0], x + j + Q[k, 1] + mvs1[y, x, 1], 0] * I[y + i, x + j, 1] 
                r2 += I[y + i + Q[k, 0] + mvs2[y, x, 0], x + j + Q[k, 1] + mvs2[y, x, 1], 2] * I[y + i, x + j, 1] 
        c[y - b, x - b, k]  = r1  
        c[y - b, x - b, k + Q.shape[0]]  = r2 
        r1 = r2 = 0

@cuda.jit
def cuda_c(I, Q, ac_off, b, c):
    y, x = cuda.grid(2)
    r1 = 0
    r2 = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    for k in range(0, Q.shape[0]):
        for i in range(-ac_off, ac_off + 1):
            for j in range(-ac_off, ac_off + 1):
                r1 += I[y + i + Q[k, 0], x + j + Q[k, 1], 0] * I[y + i, x + j, 1] 
                r2 += I[y + i + Q[k, 0], x + j + Q[k, 1], 2] * I[y + i, x + j, 1] 
        c[y - b, x - b, k]  = r1  
        c[y - b, x - b, k + Q.shape[0]]  = r2 
        r1 = r2 = 0

@cuda.jit
def cuda_C_m(I, Q, ac_off, mvs1, mvs2, b, C):
    y, x = cuda.grid(2)
    r1 = 0
    r2 = 0
    r3 = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    
    for k in range(0, Q.shape[0]):
        for l in range(0, Q.shape[0]):
            for i in range(-ac_off, ac_off + 1):
                for j in range(-ac_off, ac_off + 1):
                    r1 += I[y + i + Q[k, 0] + mvs1[y, x, 0], x + j + Q[k, 1] + mvs1[y, x, 1], 0] * \
                          I[y + i + Q[l, 0] + mvs1[y, x, 0], x + j + Q[l, 1] + mvs1[y, x, 1], 0] 

                    r2 += I[y + i + Q[k, 0] + mvs1[y, x, 0], x + j + Q[k, 1] + mvs1[y, x, 1], 0] * \
                          I[y + i + Q[l, 0] + mvs2[y, x, 0], x + j + Q[l, 1] + mvs2[y, x, 1], 2] 

                    r3 += I[y + i + Q[k, 0] + mvs2[y, x, 0], x + j + Q[k, 1] + mvs2[y, x, 1], 2] *\
                          I[y + i + Q[l, 0] + mvs2[y, x, 0], x + j + Q[l, 1] + mvs2[y, x, 1], 2]

            C[y - b, x - b, k, l] = C[y - b, x - b, l, k]   = r1
            C[y - b, x - b, k, l + Q.shape[0]] =  C[y - b, x - b, l + Q.shape[0], k] = r2
            C[y - b, x - b, k + Q.shape[0], l + Q.shape[0]] = C[y - b, x - b, l + Q.shape[0], k + Q.shape[0]] = r3
            r1 = r2 = r3 = 0

@cuda.jit
def cuda_C(I, Q, ac_off, b, C):
    y, x = cuda.grid(2)
    r1 = 0
    r2 = 0
    r3 = 0
    if(y + b >= I.shape[0] or x + b >= I.shape[1]):
        return None
    if(y < b or x < b):
        return None
    
    for k in range(0, Q.shape[0]):
        for l in range(0, Q.shape[0]):
            for i in range(-ac_off, ac_off + 1):
                for j in range(-ac_off, ac_off + 1):
                    r1 += I[y + i + Q[k, 0], x + j + Q[k, 1], 0] * \
                          I[y + i + Q[l, 0], x + j + Q[l, 1], 0] 

                    r2 += I[y + i + Q[k, 0], x + j + Q[k, 1], 0] * \
                          I[y + i + Q[l, 0], x + j + Q[l, 1], 2] 

                    r3 += I[y + i + Q[k, 0], x + j + Q[k, 1], 2] *\
                          I[y + i + Q[l, 0], x + j + Q[l, 1], 2]

            C[y - b, x - b, k, l] = C[y - b, x - b, l, k]   = r1
            C[y - b, x - b, k, l + Q.shape[0]] =  C[y - b, x - b, l + Q.shape[0], k] = r2
            C[y - b, x - b, k + Q.shape[0], l + Q.shape[0]] = C[y - b, x - b, l + Q.shape[0], k + Q.shape[0]] = r3
            r1 = r2 = r3 = 0

def get_c_m(I, Q, ac_off, mvs1, mvs2, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    c = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * Q.shape[0]))
    cuda_c_m[BPG, TPB](I, Q, ac_off, mvs1, mvs2, b, c)
    return c

def get_C_m(I, Q, ac_off, mvs1, mvs2, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    C = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * Q.shape[0], 2 * Q.shape[0])).astype(np.uint32)
    print("C array size ", C.nbytes/(10 **9) ,"Gb")
    cuda_C_m[BPG, TPB](I, Q, ac_off, mvs1, mvs2, b, C)
    return C


def get_c(I, Q, ac_off, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    c = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * Q.shape[0])).astype(np.uint32)
    cuda_c[BPG, TPB](I, Q, ac_off, b, c)
    return c

def get_C(I, Q, ac_off, b):
    TPB = (16, 16)
    bpg_y = int(np.ceil((I.shape[0] )/TPB[0]))
    bpg_x = int(np.ceil((I.shape[1] )/TPB[1]))
    BPG = (bpg_y, bpg_x)
    C = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * Q.shape[0], 2 * Q.shape[0])).astype(np.uint32)
    cuda_C[BPG, TPB](I, Q, ac_off, b, C)
    return C