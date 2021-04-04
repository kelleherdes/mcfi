import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os 
import sys
from numba import jit, typed, types
import time
from tqdm import tqdm
from scipy import signal
from sep_auto import generate_toeplitz, ac_tensor_uni
from motion import motion_est, motion_est2
import math
from cuda_auto_uni import get_c, get_C, get_c_m, get_C_m

def edge_detect(image):
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    image_x = signal.convolve2d(image, sobel, mode = 'same')
    image_y = signal.convolve2d(image, sobel.T, mode = 'same')
    mag = np.sqrt(image_x ** 2 + image_y ** 2)
    mag[mag > 255] = 255
    mag[mag < 0] = 0
    return mag.astype(np.uint8)

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

def estimate_coefficients_motion2(c_array, C_array):  
    A = np.zeros(c_array.shape)
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = np.linalg.lstsq((C_array[i, j]).astype(np.float32), c_array[i,j].astype(np.float32))[0]
    return A

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
def estimate_frame_motion(I1, I2, A, A2, k_width, k_off, b, motion, mvs1, mvs2, mvs):
    #p is past, f is future
    #k = 2
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p = A[i, j].reshape((k_width, k_width)) 
                kernel_f = A2[i, j].reshape((k_width, k_width)) 
                k_sum = np.sum(kernel_p + kernel_f)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                   kernel_f = kernel_f/k_sum
                
                y1 = i + b + mvs[i + b, j + b, 0]
                x1 = j + b + mvs[i + b, j + b, 1]
                y2 = i + b - mvs[i + b, j + b, 0]
                x2 = j + b - mvs[i + b, j + b, 1]
           
                patch1 = I1[y1 - k_off: y1 + k_off + 1, x1 - k_off: x1 + k_off + 1, c] 
                patch2 = I2[y2 - k_off: y2 + k_off + 1, x2 - k_off: x2 + k_off + 1, c]
                mask = kernel_p * patch1 + kernel_f * patch2
                predicted[i, j, c] = np.sum(mask)

    return predicted

def predict_frame_uni(image1, image2, k_width, ac_block, motion):
    #constants
    k_off  = int(k_width  / 2)
    max_motion = 10
    b      = int(ac_block / 2) + int(k_width / 2) + max_motion
    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)

    Q = generate_Q_auto(k_width)

    i_g = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    i_g[:, :, 0] = I1[:, :, 1]
    i_g[:, :, 1] = I2[:, :, 1]

    # i_r = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_r[:, :, 0] = I1[:, :, 0]
    # i_r[:, :, 1] = I2[:, :, 0]

    # i_b = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_b[:, :, 0] = I1[:, :, 2]
    # i_b[:, :, 1] = I2[:, :, 2]

    i_g2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    i_g2[:, :, 1] = I1[:, :, 1]
    i_g2[:, :, 0] = I2[:, :, 1]

    # i_r2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_r2[:, :, 1] = I1[:, :, 0]
    # i_r2[:, :, 0] = I2[:, :, 0]

    # i_b2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_b2[:, :, 1] = I1[:, :, 2]
    # i_b2[:, :, 0] = I2[:, :, 2]

    # i_g_edge = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_g_edge[:, :, 0] = edge_detect(i_g[:, :, 0])
    # i_g_edge[:, :, 1] = edge_detect(i_g[:, :, 1])

    # i_g_edge2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    # i_g_edge2[:, :, 1] = edge_detect(i_g[:, :, 0])
    # i_g_edge2[:, :, 0] = edge_detect(i_g[:, :, 1])

    # i_g_edge[i_g_edge > 255] = 255
    # i_g_edge2[i_g_edge2 > 255] = 255

    
    if(motion == 1):
        mvs1 = motion_est(i_g[:, :, 0], i_g[:, :, 1], b, max_motion)
        mvs2 = motion_est(i_g[:, :, 1], i_g[:, :, 0], b, max_motion)
        mvs = motion_est2(i_g[:, :, 0], i_g[:, :, 1], b, max_motion)
        print("Motion estimated")
        
    else:
        mvs1 = np.zeros((i_g.shape[0], i_g.shape[1], 2), dtype = np.int64)
        mvs2 = np.zeros((i_g.shape[0], i_g.shape[1], 2), dtype = np.int64)
        mvs =  np.zeros((i_g.shape[0], i_g.shape[1], 2), dtype = np.int64)
    
    c_g = get_c_m(i_g, Q, int(ac_block/2), mvs1, b)
    C_g = get_C_m(i_g, Q, int(ac_block/2), mvs1, b)
    A_g = estimate_coefficients_motion2(c_g, C_g)

    # c_r = get_c_m(i_r, Q, int(ac_block/2), mvs1, b)
    # C_r = get_C_m(i_r, Q, int(ac_block/2), mvs1, b)
    # A_r = estimate_coefficients_motion2(c_r, C_r)

    # c_b = get_c_m(i_b, Q, int(ac_block/2), mvs1, b)
    # C_b = get_C_m(i_b, Q, int(ac_block/2), mvs1, b)
    # A_b = estimate_coefficients_motion2(c_b, C_b)

    c_g2 = get_c_m(i_g2, Q, int(ac_block/2), mvs2, b)
    C_g2 = get_C_m(i_g2, Q, int(ac_block/2), mvs2, b)
    A_g2 = estimate_coefficients_motion2(c_g2, C_g2)

    # c_r2 = get_c_m(i_r2, Q, int(ac_block/2), mvs2, b)
    # C_r2 = get_C_m(i_r2, Q, int(ac_block/2), mvs2, b)
    # A_r2 = estimate_coefficients_motion2(c_r2, C_r2)

    # c_b2 = get_c_m(i_b2, Q, int(ac_block/2), mvs2, b)
    # C_b2 = get_C_m(i_b2, Q, int(ac_block/2), mvs2, b)
    # A_b2 = estimate_coefficients_motion2(c_b2, C_b2)

    # A1 = np.zeros((3, A_g.shape[0], A_g.shape[1], A_g.shape[2]))
    # A2 = np.zeros((3, A_g.shape[0], A_g.shape[1], A_g.shape[2]))
    # A1[0] = A_r
    # A1[1] = A_g
    # A1[2] = A_b

    # A2[0] = A_r2
    # A2[1] = A_g2
    # A2[2] = A_b2

    # c_g_edge = get_c_m(i_g_edge, Q, int(ac_block/2), mvs1, b)
    # C_g_edge = get_C_m(i_g_edge, Q, int(ac_block/2), mvs1, b)
    # Ae = estimate_coefficients_motion2(c_g_edge, C_g_edge)

    # c_g_edge2 = get_c_m(i_g_edge2, Q, int(ac_block/2), mvs2, b)
    # C_g_edge2 = get_C_m(i_g_edge2, Q, int(ac_block/2), mvs2, b)
    # Ae2 = estimate_coefficients_motion2(c_g_edge2, C_g_edge2)

    #predicted = estimate_frame_motion(I1, I2, A_g, A_g2, Ae, Ae2, k_width, k_off, b, motion, mvs1, mvs2, mvs)
    predicted = estimate_frame_motion(I1, I2, A_g, A_g2, k_width, k_off, b, motion, mvs1, mvs2, mvs)
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
        image1 = '../football/image4.png'
        image2 = '../football/image5.png'
        image3 = '../football/image6.png'
        out = '../output/out.png'
        k_width = 3
        ac_block = 19
        motion = 1
    print("Kernel size:", k_width)
    print("Autocorrelation kernel size:", ac_block)
    #kernel max offsets (the max index to be used)    
    print("Predicting frames")
    predicted = predict_frame_uni(image1, image3, k_width, ac_block, motion)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()