import numpy as np
import cv2
import csv
import os 
import sys
from numba import jit, typed, types
import time
from tqdm import tqdm
from scipy import signal
from sep_auto import generate_toeplitz, ac_tensor_uni
from motion import motion_est
import math

def q_dict(Q):
    qd = typed.Dict.empty(key_type = types.UniTuple(types.int64, 2), value_type=types.int64, )
    for i in range(Q.shape[0]):
        key = np.asarray([Q[i, 0], Q[i, 1]], dtype = np.int64)
        qd[tuple(key)] = i
    return qd

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

def generate_Q_auto(width):
    Q = np.zeros((width, width, 2))
    #maximum offset
    off = int(width / 2)
    #e.g for a 3x3 kernel centered at current pixel max offset is 
    for i in range(0, width):
        Q[i, :, 0] = i - off
        Q[:, i, 1] = i - off
    Q = Q.reshape((width * width, 2)).astype(np.int64)
    return Q

def generate_Q(width):
    Q = np.zeros((2 * width - 1, 2))
    #maximum offset
    off = int(width / 2)
    #generate 1d array of offset vectors
    #for kernel width of 3 should be [-1,0], [0,0], [1,0], [0, -1], [0,0] -- plus shape without repeating origin
    for i in range(0, width):
        Q[i] = np.array([i - off, 0])

    for i in range(0, off):
        Q[width + i] = np.array([0, i - off])
    
    for i in range(0, off):
        Q[width + off + i] = np.array([0, i + 1])
    Q = Q.astype(np.int64)
    return Q

@jit(nopython=True)
def calc_a_motion(x1, x2, Q, q_motion, toeplitz, autocor, q_d, q_md, k_width, mvs):
    #autocorrelation
    q_len = 2 * k_width - 1
    c = np.zeros(q_len)
    C = np.zeros((q_len, q_len))
    for i in range(q_len):
        c[i] = autocor[x1, x2, q_md[(int(q_motion[i, 0] - mvs[x1, x2, 0]), int(q_motion[i, 1] - mvs[x1, x2, 1]))]]
        for j in range(q_len):
            offset = Q[j] - Q[i]
            C[i, j] = toeplitz[int(x1 + Q[i, 0] - mvs[x1, x2, 0]), int(x2 + Q[i, 1] - mvs[x1, x2, 1]), q_d[(offset[0], offset[1])]]
    a = np.linalg.lstsq(C.astype(np.float32), c.astype(np.float32))[0]   
    return a

@jit(nopython=True)
def calc_a(x1, x2, Q, toeplitz, autocor, q_d, k_width):
    #autocorrelation
    c = autocor[x1, x2]
    q_len = 2 * k_width - 1
    C = np.zeros((q_len, q_len))
    for i in range(q_len):
        for j in range(q_len):
            offset = Q[j] - Q[i]
            C[i, j] = toeplitz[x1 + Q[i, 0], x2 + Q[i, 1], q_d[(offset[0], offset[1])]]
    a = np.linalg.lstsq(C.astype(np.float32), c.astype(np.float32))[0]   
    return a

@jit(nopython=True)
def estimate_frame_motion(I1, I2, A, k_width, k_off, b, motion, mvs):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    y = 0
    x = 0
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                   = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1]
                kernel_p[:, k_off]          = a_p[ 0 : k_width]
                kernel_p[k_off, 0 : k_off ] = a_p[k_width : k_width + k_off]
                kernel_p[k_off, k_off + 1:] = a_p[k_width + k_off:]
                k_sum = np.sum(kernel_p)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                
                y = i + b
                x = j + b
                if(motion == 1):
                    y -= mvs[i + b, j + b, 0]
                    x -= mvs[i + b, j + b, 1]
           
                patch = I1[y - k_off: y + k_off + 1, x - k_off: x + k_off + 1, channel] 
                mask = kernel_p * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted

@jit(nopython=True)
def estimate_frame(I1, I2, A, k_width, k_off, b):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    y = 0
    x = 0
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                   = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1]
                kernel_p[:, k_off]          = a_p[ 0 : k_width]
                kernel_p[k_off, 0 : k_off ] = a_p[k_width : k_width + k_off]
                kernel_p[k_off, k_off + 1:] = a_p[k_width + k_off:]
                k_sum = np.sum(kernel_p)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                
                y = i + b
                x = j + b
                patch = I1[y - k_off: y + k_off + 1, x - k_off: x + k_off + 1, channel] 
                mask = kernel_p * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted
def estimate_coefficients(I, toeplitz, autocor, Q, q_d, k_width, b):  
    A = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * k_width - 1))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = calc_a(i + b, j + b, Q, toeplitz, autocor, q_d, k_width)
    return A

def estimate_coefficients_motion(I, toeplitz, autocor, Q, q_motion, q_d, qm_d, k_width, b, mvs):  
    A = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * k_width - 1))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = calc_a_motion(i + b, j + b, Q, q_motion, toeplitz, autocor, q_d, qm_d, k_width, mvs)
    return A

def predict_frame_uni(image1, image2, k_width, ac_block, motion):
    #constants
    k_off  = int(k_width  / 2)
    max_motion = 10
    b      = int(ac_block / 2) + int(k_width / 2) + max_motion
    ac_size = k_width * int(1 + max_motion/k_off) - 1
    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    Q = generate_Q(k_width)
    double_Q = generate_Q_auto(2 * k_width - 1)
    q_d = q_dict(double_Q)
    i1 = I1[:, :, 1]
    i2 = I2[:, :, 1]
    if(motion == 1):
        mvs = motion_est(i1, i2, b, max_motion)
        q_motion = Q
        qm_d = q_dict(generate_Q_auto(ac_size))
    else:
        mvs = 0

    print("Generating toeplitzs")
    toeplitz = generate_toeplitz(i1, i1, ac_block, k_width, double_Q, b)
    if(motion):
        autocor  = ac_tensor_uni(i1, i2, ac_block, k_width, generate_Q_auto(ac_size), b)
        A = estimate_coefficients_motion(I2[:, :, 1], toeplitz, autocor, Q, q_motion, q_d, qm_d, k_width, b, mvs)
        predicted = estimate_frame_motion(I1, I2, A, k_width, k_off, b, motion, mvs)
    else:
        autocor  = ac_tensor_uni(i1, i2, ac_block, k_width, Q, b)
        A = estimate_coefficients(I2[:, :, 1], toeplitz, autocor, Q, q_d, k_width, b)
        predicted = estimate_frame(I1, I2, A, k_width, k_off, b)

    # with open('a.npy', 'wb') as a_file:
    #     np.save(a_file, A)
    predicted[predicted > 255] = 255
    predicted[predicted < 0] = 0
    return predicted

#global variables
def main():
    if(len(sys.argv) == 7):
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        out = sys.argv[3]
        k_width = int(sys.argv[4])
        ac_block = int(sys.argv[5])
        motion = int(sys.arg[6])
    
    else:
        image1 = '../images/image0.png'
        image2 = '../images/image1.png'
        out = '../output/out.png'
        k_width = 5
        ac_block = 13
        motion = 1
    print("Kernel size:", k_width)
    print("Autocorrelation kernel size:", ac_block)
    #kernel max offsets (the max index to be used)    
    print("Predicting frames")
    predicted = predict_frame_uni(image1, image2, k_width, ac_block, motion)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()