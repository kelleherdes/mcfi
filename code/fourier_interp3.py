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
from cuda_fourier import get_c_m, get_C_m
import scipy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
from numpy import ma
from matplotlib import cbook

class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

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
def estimate_frame(I1, I2, A, A2, b, mvs, k_width):
    k_off = int(k_width / 2)
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p = A[i, j,  c].reshape((k_width, k_width))
                kernel_f = A2[i, j, c].reshape((k_width, k_width))
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

def diag_A(A, mvs, max_motion, k_width, b):
    k_w2 = int(2 * max_motion) + int(k_width)
    cen = int(k_w2/2)
    k_off = int(k_width / 2)
    A_star = np.zeros((A.shape[0], A.shape[1], k_w2, k_w2))
    print("Calculating new kernels")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A_time = np.zeros((k_w2, k_w2))
            A_time[cen - k_off + mvs[i + b, j + b, 0] : cen + k_off + 1 + mvs[i + b, j + b, 0], \
                   cen - k_off + mvs[i + b, j + b, 1] : cen + k_off + 1 + mvs[i + b, j + b, 1]] = A[i, j].reshape((k_width, k_width))
            A_freq = fft.fft(A_time)
            D, P = np.linalg.eig(A_freq)
            D = D * np.eye(k_w2)
            P_inv = np.linalg.inv(P)
            A_star[i, j] = fft.ifft(P @ np.sqrt(D) @ P_inv)
    return A_star

def predict_frame_uni(image1, image2, k_width, ac_block):
    #constants
    import cv2
    max_motion = 10
    b      = int(ac_block / 2) + int(k_width / 2) + max_motion

    I1 = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2YUV), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2YUV), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)

    Q = generate_Q_auto(k_width)

    iy = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    iu = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    iv = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)

    iy2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    iu2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)
    iv2 = np.zeros((I1.shape[0], I1.shape[1], 2), dtype = np.int32)

    iy[:, :, 0] = I1[:, :, 0]
    iu[:, :, 0] = I1[:, :, 1]
    iv[:, :, 0] = I1[:, :, 2]

    iy[:, :, 1] = I2[:, :, 0]
    iu[:, :, 1] = I2[:, :, 1]
    iv[:, :, 1] = I2[:, :, 2]

    iy2[:, :, 0] = I2[:, :, 0]
    iu2[:, :, 0] = I2[:, :, 1]
    iv2[:, :, 0] = I2[:, :, 2]

    iy2[:, :, 1] = I1[:, :, 0]
    iu2[:, :, 1] = I1[:, :, 1]
    iv2[:, :, 1] = I1[:, :, 2]

    mvs = motion_est2(iy[:, :, 0], iy[:, :, 1], b, max_motion)
    
    cy = get_c_m(iy, Q, int(ac_block/2), mvs, b)
    Cy = get_C_m(iy, Q, int(ac_block/2), mvs, b)
    Ay = estimate_coefficients_motion2(cy, Cy)

    cu = get_c_m(iu, Q, int(ac_block/2), mvs, b)
    Cu = get_C_m(iu, Q, int(ac_block/2), mvs, b)
    Au = estimate_coefficients_motion2(cu, Cu)

    cv = get_c_m(iv, Q, int(ac_block/2), mvs, b)
    Cv = get_C_m(iv, Q, int(ac_block/2), mvs, b)
    Av = estimate_coefficients_motion2(cv, Cv)
    ###################
    cy2 = get_c_m(iy2, Q, int(ac_block/2), -mvs, b)
    Cy2 = get_C_m(iy2, Q, int(ac_block/2), -mvs, b)
    Ay2 = estimate_coefficients_motion2(cy2, Cy2)

    cu2 = get_c_m(iu2, Q, int(ac_block/2), -mvs, b)
    Cu2 = get_C_m(iu2, Q, int(ac_block/2), -mvs, b)
    Au2 = estimate_coefficients_motion2(cu2, Cu2)

    cv2 = get_c_m(iv2, Q, int(ac_block/2), -mvs, b)
    Cv2 = get_C_m(iv2, Q, int(ac_block/2), -mvs, b)
    Av2 = estimate_coefficients_motion2(cv2, Cv2)

    A = np.zeros((Ay.shape[0], Ay.shape[1], 3, Ay.shape[2]))
    A[..., 0, :] = Ay
    A[..., 1, :] = Au
    A[..., 2, :] = Av

    A2 = np.zeros((Ay.shape[0], Ay.shape[1], 3, Ay.shape[2]))
    A2[..., 0, :] = Ay2
    A2[..., 1, :] = Au2
    A2[..., 2, :] = Av2

    predicted = estimate_frame(I1, I2, A, A2, b, mvs, k_width)
    predicted[predicted > 255] = 255
    predicted[predicted < 0] = 0

    import cv2
    predicted = cv2.cvtColor(predicted.astype(np.uint8), cv2.COLOR_YUV2BGR)
    return predicted.astype(np.uint8)

#global variables
def main():
    if(len(sys.argv) == 8):
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        image3 = sys.argv[3]
        out = sys.argv[4]
        k_width = int(sys.argv[5])
        ac_block = int(sys.argv[6])

    else:
        image1 = '../images/image0.png'
        image2 = '../images/image1.png'
        image3 = '../images/image2.png'
        out = '../output/out.png'
        k_width = 3
        ac_block = 5
    
    print("Kernel size:", k_width)
    print("Autocorrelation kernel size:", ac_block)
    #kernel max offsets (the max index to be used)    
    print("Predicting frames")
    predicted = predict_frame_uni(image1, image3, k_width, ac_block)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()