import numpy as np
import cv2
import csv
import os 
import sys
from numba import jit, typed, types
import time
from tqdm import tqdm
from scipy import signal
from motion import motion_est
import math


    
@jit(nopython=True)
def simple_kernel(image1, image2, x, k_width, ac_block, Q):
    k_size = k_width ** 2
    off = int(ac_block / 2)
    C = np.zeros((k_size, k_size))
    c = np.zeros(k_size)
    for i in range(0, k_size):
        c[i] = np.sum(image2[x[0]           - off : x[0]           + off + 1, x[1]           - off : x[1]           + off + 1] * \
               image1[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1])
        for j in range(i, k_size):
            C[j, i] = C[i, j] = np.sum(image1[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1] * \
                                image1[x[0] + Q[j, 0] - off : x[0] + Q[j, 0] + off + 1, x[1] + Q[j, 1] - off : x[1] + Q[j, 1] + off + 1])

    a = np.linalg.lstsq(C, c)[0]
    return a

@jit(nopython=True)
def simple_kernel2(image1, image2, image3, x, k_width, ac_block, Q):
    k_size = k_width ** 2
    off = int(ac_block / 2)
    C = np.zeros((2 * k_size, 2 * k_size), dtype=np.float32)
    c = np.zeros(2 * k_size, dtype = np.float32)
    for i in range(0, k_size):
        c[i] = \
            np.sum(image2[x[0]           - off : x[0]           + off + 1, x[1]           - off : x[1]           + off + 1] * \
                   image1[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1])

        c[i + k_size] = \
            np.sum(image2[x[0]           - off : x[0]           + off + 1, x[1]           - off : x[1]           + off + 1] * \
                   image3[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1])

        for j in range(0, k_size):
            C[i, j] = \
                np.sum(image1[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1] * \
                       image1[x[0] + Q[j, 0] - off : x[0] + Q[j, 0] + off + 1, x[1] + Q[j, 1] - off : x[1] + Q[j, 1] + off + 1])
            
            C[i, j + k_size] = C[j + k_size, i] = \
                np.sum(image1[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1] * \
                       image3[x[0] + Q[j, 0] - off : x[0] + Q[j, 0] + off + 1, x[1] + Q[j, 1] - off : x[1] + Q[j, 1] + off + 1])

            C[i + k_size, j + k_size] = \
                np.sum(image3[x[0] + Q[i, 0] - off : x[0] + Q[i, 0] + off + 1, x[1] + Q[i, 1] - off : x[1] + Q[i, 1] + off + 1] * \
                       image3[x[0] + Q[j, 0] - off : x[0] + Q[j, 0] + off + 1, x[1] + Q[j, 1] - off : x[1] + Q[j, 1] + off + 1])

    a = np.linalg.lstsq(C, c)[0]
    return a


def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

def get_mse_reduced(niklaus, a):
    N = np.count_nonzero(niklaus)
    mse = np.sum((niklaus - a) ** 2)/N
    return mse

def get_mse(image1, image2):
    assert (image1.shape == image2.shape)
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    return mse

def generate_Q(width):
    Q = np.zeros((width, width, 2))
    #maximum offset
    off = int(width / 2)
    #e.g for a 3x3 kernel centered at current pixel max offset is 
    for i in range(0, width):
        Q[i, :, 0] = i - off
        Q[:, i, 1] = i - off
    Q = Q.reshape((width * width, 2)).astype(np.int64)
    return Q

@jit(nopython=True)
def estimate_frame2(I1, I3, A, k_width, b):
    k_off = int(k_width / 2)
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))

    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel = A[i, j, :].reshape((2 * k_width,  k_width))
                k_sum = np.sum(kernel)
                if(k_sum != 0):
                   kernel = kernel/np.sum(kernel)
                patch1 = I1[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 , c] 
                patch2 = I3[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 , c]
                patch = np.concatenate((patch1, patch2), axis = 0)
                mask = kernel * patch
                predicted[i, j, c] = np.sum(mask)
    return predicted


def estimate_coefficients(image1, image2, Q, k_width, ac_block, b):  
    A = np.zeros((image1.shape[0] - 2 * b, image2.shape[1] - 2 * b, k_width * k_width))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = simple_kernel(image1, image2, np.array([i + b, j + b]).astype(np.int64), k_width, ac_block, Q)
    return A

def estimate_coefficients2(image1, image2,image3, Q, k_width, ac_block, b):  
    A = np.zeros((image1.shape[0] - 2 * b, image2.shape[1] - 2 * b, 2 * k_width * k_width))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = simple_kernel2(image1, image2, image3, np.array([i + b, j + b]).astype(np.int64), k_width, ac_block, Q)
    return A

#global variables
def main():
    if(len(sys.argv) == 7):
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        out = sys.argv[3]
        k_width = int(sys.argv[4])
        ac_block = int(sys.argv[5])
    
    else:
        image1 = '../images/image0.png'
        image2 = '../images/image1.png'
        image3 = '../images/image2.png'
        out = '../output/simple_out.png'
        k_width = 3
        ac_block = 11

    b = ac_block
    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    Q = generate_Q(k_width)
    A = estimate_coefficients2(I1[:, :, 1], I2[:, :, 1], I3[:, :, 1], Q, k_width, ac_block, b)
    predicted = estimate_frame2(I1, I3, A, k_width, b)
    print(predicted.dtype)
    print(cv2.imread(image2).dtype)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()