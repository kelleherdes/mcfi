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
import math
from cuda_auto import get_c_m, get_C_m, get_c, get_C

@jit(nopython=True)
def reverse_points(mvs):
    reverse_mvs = np.zeros(mvs.shape)
    for i in range(mvs.shape[0]):
        for j in range(mvs.shape[1]):
            reverse_mvs[i - mvs[i, j, 0], j - mvs[i, j, 1]] = mvs[i, j] 
    return reverse_mvs.astype(np.int64)

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

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

def generate_Q2(width):
    q_len = 2 * width - 1
    Q = np.zeros((2 * q_len, 3))
    #maximum offset
    off = int(width / 2)
    #generate 1d array of offset vectors
    #for kernel width of 3 should be [-1,0], [0,0], [1,0], [0, -1], [0,0] -- plus shape without repeating origin
    for i in range(0, width):
        Q[i] = np.array([i - off, 0, -1])
        Q[i + q_len] = np.array([i - off, 0, 1])

    for i in range(0, off):
        Q[width + i] = np.array([0, i - off, -1])
        Q[width + i + q_len] = np.array([0, i - off, 1])
    
    for i in range(0, off):
        Q[width + off + i] = np.array([0, i + 1, -1])
        Q[width + off + i + q_len] = np.array([0, i + 1, 1])
    Q = Q.astype(np.int64)
    return Q

def generate_Q(width):
    q_len = 2 * width - 1
    Q = np.zeros((q_len, 2))
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
def estimate_frame(I1, I2, A, k_width, k_off, b):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                    = np.zeros((k_width, k_width))
                kernel_f                    = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1, c]
                a_f                         = A[i, j, 2 * k_width - 1:, c]
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

                patch1 = I1[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1, c] 
                patch2 = I2[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1, c]
                patch = np.concatenate((patch1, patch2), axis = 0)
                kernel = np.concatenate((kernel_p, kernel_f), axis = 0)
                mask = kernel * patch
                predicted[i, j, c] = np.sum(mask)
    return predicted

@jit(nopython=True)
def estimate_frame_motion(I1, I2, A, k_width, k_off, b, mvs1, mvs2):
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                    = np.zeros((k_width, k_width))
                kernel_f                    = np.zeros((k_width, k_width))
                a_p                         = A[i, j, 0 : 2 * k_width - 1, c]
                a_f                         = A[i, j, 2 * k_width - 1:, c]
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
                y1 = i + b + mvs1[i + b, j + b, 0]
                x1 = j + b + mvs1[i + b, j + b, 1]
                y2 = i + b + mvs2[i + b, j + b, 0]
                x2 = j + b + mvs2[i + b, j + b, 1]
                patch1 = I1[y1 - k_off: y1 + k_off + 1, x1 - k_off: x1 + k_off + 1, c] 
                patch2 = I2[y2 - k_off: y2 + k_off + 1, x2 - k_off: x2 + k_off + 1, c]
                patch = np.concatenate((patch1, patch2), axis = 0)
                kernel = np.concatenate((kernel_p, kernel_f), axis = 0)
                mask = kernel * patch
                predicted[i, j, c] = np.sum(mask)
    return predicted


def estimate_coefficients_motion(c_array, C_array):  
    A = np.zeros(c_array.shape)
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = np.linalg.lstsq((C_array[i, j]).astype(np.float32), c_array[i,j].astype(np.float32))[0]
    return A



def predict_frame(image1, image2, image3, k_width, ac_block, motion):

    k_off  = int(k_width  / 2)
    max_motion = 10
    b      = int(ac_block / 2) + int(k_width / 2) + max_motion

    # I1 = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2YUV), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    # I2 = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2YUV), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    # I3 = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(image3), cv2.COLOR_BGR2YUV), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)

    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)

    Q = generate_Q(k_width)
    i_r = np.zeros((I1.shape[0], I1.shape[1], 3), dtype = np.int32)
    i_r[:, :, 0] = I1[:, :, 0]
    i_r[:, :, 1] = I2[:, :, 0]
    i_r[:, :, 2] = I3[:, :, 0]

    i_g = np.zeros((I1.shape[0], I1.shape[1], 3), dtype = np.int32)
    i_g[:, :, 0] = I1[:, :, 1]
    i_g[:, :, 1] = I2[:, :, 1]
    i_g[:, :, 2] = I3[:, :, 1]

    i_b = np.zeros((I1.shape[0], I1.shape[1], 3), dtype = np.int32)
    i_b[:, :, 0] = I1[:, :, 2]
    i_b[:, :, 1] = I2[:, :, 2]
    i_b[:, :, 2] = I3[:, :, 2]

    if(motion == 1):
        mvs1 = motion_est(i_r[:, :, 0], i_r[:, :, 1], b, max_motion)
        mvs2 = motion_est(i_r[:, :, 2], i_r[:, :, 1], b, max_motion)

        c_r = get_c_m(i_r, Q, int(ac_block/2), mvs1, mvs2, b)
        c_g = get_c_m(i_g, Q, int(ac_block/2), mvs1, mvs2, b)
        c_b = get_c_m(i_b, Q, int(ac_block/2), mvs1, mvs2, b)
        C_r = get_C_m(i_r, Q, int(ac_block/2), mvs1, mvs2, b)
        C_g = get_C_m(i_g, Q, int(ac_block/2), mvs1, mvs2, b)
        C_b = get_C_m(i_b, Q, int(ac_block/2), mvs1, mvs2, b)
        A_r = estimate_coefficients_motion(c_r, C_r)
        A_g = estimate_coefficients_motion(c_g, C_g)
        A_b = estimate_coefficients_motion(c_b, C_b)
        A = np.zeros((A_r.shape[0], A_r.shape[1], A_r.shape[2], 3))
        A[:, :, :, 0] = A_r
        A[:, :, :, 1] = A_g
        A[:, :, :, 2] = A_b
        predicted = estimate_frame_motion(I1, I3, A, k_width, k_off, b, mvs1, mvs2)

    else:
        #c_array = get_c
        c_r = get_c(i_r, Q, int(ac_block/2), b)
        c_g = get_c(i_g, Q, int(ac_block/2), b)
        c_b = get_c(i_b, Q, int(ac_block/2), b)

        C_r = get_C(i_r, Q, int(ac_block/2), b)
        C_g = get_C(i_g, Q, int(ac_block/2), b)
        C_b = get_C(i_b, Q, int(ac_block/2), b)

        A_r = estimate_coefficients_motion(c_r, C_r)
        A_g = estimate_coefficients_motion(c_g, C_g)
        A_b = estimate_coefficients_motion(c_b, C_b)

        A = np.zeros((A_y.shape[0], A_y.shape[1], A_y.shape[2], 3))
        A[:,:,:,0] = A_r
        A[:,:,:,1] = A_g
        A[:,:,:,2] = A_b
        predicted = estimate_frame(I1, I3, A, k_width, k_off, b)

    #predicted = cv2.cvtColor(predicted.astype(np.uint8), cv2.COLOR_YUV2BGR)
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
        motion = int(sys.argv[7])
    
    else:
        image1 = '../images/image0.png'
        image2 = '../images/image1.png'
        image3 = '../images/image2.png'
        out = '../output/out.png'
        k_width = 3
        ac_block = 11
        motion = 1
    print("Kernel size:", k_width)
    print("Autocorrelation kernel size:", ac_block)
    #kernel max offsets (the max index to be used)
    
    print("Predicting frames")
    predicted = predict_frame(image1, image2, image3, k_width, ac_block, motion)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()