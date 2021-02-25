import numpy as np
import cv2
import csv
import os 
import sys
from numba import jit, typed, types
import time
from tqdm import tqdm
from scipy import signal
from sep_auto import generate_toeplitz, generate_ac_tensor, ac_tensor_uni
from motion import motion_est

def q_dict(Q):
    qd = typed.Dict.empty(key_type = types.UniTuple(types.int64, 2), value_type=types.int64, )
    for i in range(Q.shape[0]):
        key = np.asarray([Q[i, 0], Q[i, 1]], dtype = np.int64)
        qd[tuple(key)] = i
    return qd

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
def calc_a(x1, x2, Q, toeplitz1, toeplitz2, toeplitz3, autocor, q_d, k_width):
    #autocorrelation
    c = autocor[x1, x2]
    q_len = 2 * k_width - 1
    C = np.zeros((2 * q_len, 2 * q_len))
    for i in range(q_len):
        for j in range(q_len):
            offset = Q[j] - Q[i]
            C[i, j]                           = toeplitz1[x1 + Q[i, 0], x2 + Q[i, 1], q_d[(offset[0], offset[1])]]
            C[j, i + q_len] = C[i + q_len, j] = toeplitz2[x1 + Q[i, 0], x2 + Q[i, 1], q_d[(offset[0], offset[1])]]
            C[i + q_len, j + q_len]           = toeplitz3[x1 + Q[i, 0], x2 + Q[i, 1], q_d[(offset[0], offset[1])]]
    a = np.linalg.lstsq(C.astype(np.float32), c.astype(np.float32))[0]   
    #print(a.shape)
    return a

@jit(nopython=True)
def calc_a_motion(x1, x2, Q, toeplitz1, toeplitz2, toeplitz3, autocor1, autocor2, q_d, qm_d, qt_d, k_width, mvs1, mvs2, mvs3):
    #autocorrelation
    q_len = 2 * k_width - 1
    c = np.zeros(2 * q_len)
    C = np.zeros((2 * q_len, 2 * q_len))
    for i in range(q_len):
        c[i] =         autocor1[x1, x2, qm_d[(int(Q[i, 0] - mvs1[x1, x2, 0]), int(Q[i, 1] - mvs1[x1, x2, 1]))]]
        c[i + q_len] = autocor2[x1, x2, qm_d[(int(Q[i, 0] - mvs2[x1, x2, 0]), int(Q[i, 1] - mvs2[x1, x2, 1]))]]
        for j in range(q_len):
            offset = Q[j] - Q[i]
            C[i, j]                           = toeplitz1[x1 + Q[i, 0] - mvs1[x1, x2, 0], x2 + Q[i, 1] - mvs1[x1, x2, 1], q_d[(offset[0], offset[1])]]
            C[j, i + q_len] = C[i + q_len, j] = toeplitz2[x1 + Q[i, 0] - mvs1[x1, x2, 0], x2 + Q[i, 1] - mvs1[x1, x2, 1], qt_d[(int(offset[0] + mvs3[x1, x2, 0]), int(offset[1] + mvs3[x1, x2, 1]))]]
            C[i + q_len, j + q_len]           = toeplitz3[x1 + Q[i, 0] - mvs2[x1, x2, 0], x2 + Q[i, 1] - mvs2[x1, x2, 1], q_d[(offset[0], offset[1])]]
    a = np.linalg.lstsq(C.astype(np.float32), c.astype(np.float32))[0]   
    return a

@jit(nopython=True)
def estimate_frame(I1, I2, A, k_width, k_off, b):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                    = np.zeros((k_width, k_width))
                kernel_f                    = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1]
                a_f                         = A[i, j, 2 * k_width - 1:]
                kernel_p[:, k_off]          = a_p[ 0 : k_width]
                kernel_p[k_off, 0 : k_off ] = a_p[k_width : k_width + k_off]
                kernel_p[k_off, k_off + 1:] = a_p[k_width + k_off:]
                kernel_f[:, k_off]          = a_f[ 0 : k_width]
                kernel_f[k_off, 0 : k_off ] = a_f[k_width : k_width + k_off]
                kernel_f[k_off, k_off + 1:] = a_f[k_width + k_off:]
                
                k_sum = np.sum(kernel_p + kernel_f)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                   kernel_f = kernel_f/k_sum

                patch1 = I1[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 ,channel] 
                patch2 = I2[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 ,channel]
                patch = np.concatenate((patch1, patch2), axis = 0)
                kernel = np.concatenate((kernel_p, kernel_f), axis = 0)
                mask = kernel * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted

@jit(nopython=True)
def estimate_frame_motion(I1, I2, A, k_width, k_off, b, mvs1, mvs2):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                    = np.zeros((k_width, k_width))
                kernel_f                    = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1]
                a_f                         = A[i, j, 2 * k_width - 1:]
                kernel_p[:, k_off]          = a_p[ 0 : k_width]
                kernel_p[k_off, 0 : k_off ] = a_p[k_width : k_width + k_off]
                kernel_p[k_off, k_off + 1:] = a_p[k_width + k_off:]
                kernel_f[:, k_off]          = a_f[ 0 : k_width]
                kernel_f[k_off, 0 : k_off ] = a_f[k_width : k_width + k_off]
                kernel_f[k_off, k_off + 1:] = a_f[k_width + k_off:]
                
                k_sum = np.sum(kernel_p + kernel_f)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                   kernel_f = kernel_f/k_sum
                y1 = i + b - mvs1[i + b, j + b, 0]
                x1 = j + b - mvs1[i + b, j + b, 1]
                y2 = i + b - mvs2[i + b, j + b, 0]
                x2 = j + b - mvs2[i + b, j + b, 1]
                patch1 = I1[y1 - k_off: y1 + k_off + 1, x1 - k_off: x1 + k_off + 1 ,channel] 
                patch2 = I2[y2 - k_off: y2 + k_off + 1, x2 - k_off: x2 + k_off + 1 ,channel]
                patch = np.concatenate((patch1, patch2), axis = 0)
                kernel = np.concatenate((kernel_p, kernel_f), axis = 0)
                mask = kernel * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted
def estimate_coefficients_motion(I, toeplitz1, toeplitz2, toeplitz3, autocor1, autocor2, Q, q_d, qm_d, qt_d, k_width, b, mvs1, mvs2, mvs3):  
    A = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 4 * k_width - 2))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = calc_a_motion(i + b, j + b, Q, toeplitz1, toeplitz2, toeplitz3, autocor1, autocor2, q_d, qm_d, qt_d, k_width, mvs1, mvs2, mvs3)
    return A

def estimate_coefficients(I, toeplitz1, toeplitz2, toeplitz3, autocor, Q, q_d, k_width, b):  
    A = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 4 * k_width - 2))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = calc_a(i + b, j + b, Q, toeplitz1, toeplitz2, toeplitz3, autocor, q_d, k_width)
    return A

def predict_frame(image1, image2, image3, k_width, ac_block, motion):

    k_off  = int(k_width  / 2)
    max_motion = 10
    b      = int(ac_block / 2) + int(k_width / 2) + max_motion
    ac_size = k_width * int(1 + max_motion/k_off) - 1

    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    Q = generate_Q(k_width)
    double_Q = generate_Q_auto(2 * k_width - 1)
    q_d = q_dict(double_Q)
    i1 = I1[:, :, 1]
    i2 = I2[:, :, 1]
    i3 = I3[:, :, 1]
    if(motion == 1):
        mvs1 = motion_est(i1, i2, b, max_motion)
        mvs2 = motion_est(i3, i2, b, max_motion)
        mvs3 = motion_est(i1, i3, b, max_motion)
        qm_d = q_dict(generate_Q_auto(ac_size))
        qt_d = q_dict(generate_Q_auto(2 * ac_size - 1))

    print("Generating toeplitzs")
    toeplitz1 = generate_toeplitz(i1, i1, ac_block, k_width, double_Q, b)
    toeplitz3 = generate_toeplitz(i3, i3, ac_block, k_width, double_Q, b)
    if(motion):
        toeplitz2 = generate_toeplitz(i3, i1, ac_block, k_width, generate_Q_auto(2 * ac_size - 1), b)
        autocor1  = ac_tensor_uni(i1, i2, ac_block, k_width, generate_Q_auto(ac_size), b)
        autocor2  = ac_tensor_uni(i3, i2, ac_block, k_width, generate_Q_auto(ac_size), b)
        A = estimate_coefficients_motion(I2[:, :, 1], toeplitz1, toeplitz2, toeplitz3, autocor1, autocor2, Q, q_d, qm_d, qt_d, k_width, b, mvs1, mvs2, mvs3)
        predicted = estimate_frame_motion(I1, I3, A,  k_width, k_off, b, mvs1, mvs2)
    else:
        toeplitz2 = generate_toeplitz(i3, i1, ac_block, k_width, double_Q, b)
        autocor2  = generate_ac_tensor(i2, i1, i3, ac_block, k_width, Q, b)
        A = estimate_coefficients(I2[:, :, 1], toeplitz1, toeplitz2, toeplitz3, autocor2, Q, q_d, k_width, b)
        predicted = estimate_frame(I1, I3, A,  k_width, k_off, b)

    with open('a.npy', 'wb') as a_file:
        np.save(a_file, A)
    predicted = 0
    predicted = estimate_frame(I1, I3, A,  k_width, k_off, b)
    predicted[predicted > 255] = 255
    predicted[predicted < 0] = 0
    return predicted

#global variables
def main():
    if(len(sys.argv) == 8):
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        image3 = sys.argv[3]
        out = sys.argv[4]
        k_width = int(sys.argv[5])
        ac_block = int(sys.argv[6])
        motion = int(sys.argv[6])
    
    else:
        image1 = '../images/image0.png'
        image2 = '../images/image1.png'
        image3 = '../images/image2.png'
        out = '../output/out.png'
        k_width = 3
        ac_block = 3
        motion = 1
    print("Kernel size:", k_width)
    print("Autocorrelation kernel size:", ac_block)
    #kernel max offsets (the max index to be used)
    
    print("Predicting frames")
    predicted = predict_frame(image1, image2, image3, k_width, ac_block, motion)
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()