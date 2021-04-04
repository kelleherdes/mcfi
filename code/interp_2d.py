import numpy as np
import cv2
import csv
import os 
import sys
from numba import jit, typed, types
import time
from tqdm import tqdm
from scipy import signal
from sep_auto import generate_toeplitz, generate_ac_tensor
import math
from motion import motion_est
from cuda_auto_2d import get_c_m, get_C_m, get_c, get_C

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

def generate_Q(width):
    Q = np.zeros((width, width, 3))
    #maximum offset
    off = int((width  - (width % 2))/2)
    #e.g for a 3x3 kernel centered at current pixel max offset is 
    for i in range(0, width):
        Q[i, :, 0] = i - off
        Q[:, i, 1] = i - off
    Q = Q.reshape((width * width, 3)).astype(np.int64)
    Q[:, 2] = -1
    return Q


@jit(nopython=True)
def make_C(x1, x2, Q, toeplitz1, toeplitz2, toeplitz3, autocor2, indices, k_size):
    C = np.zeros((2 * k_size, 2 * k_size), dtype = np.float32) 
    for i in range(0, k_size):
        for j in range (0, k_size):
            C[i, j] = toeplitz1[x1 + Q[i, 0], x2 + Q[i, 1], int(indices[i * k_size + j])]
            C[i + k_size, j] = toeplitz2[x1 + Q[i, 0], x2 + Q[i, 1], int(indices[i * k_size + j])]
            C[j, i + k_size] = C[i + k_size, j]
            C[i + k_size, j + k_size] = toeplitz3[x1 + Q[i, 0], x2 + Q[i, 1], int(indices[i * k_size + j])]
    return C

def calc_a(x1, x2, Q, toeplitz1, toeplitz2, toeplitz3, autocor2, indices, k_size):
    #autocorrelation
    C = make_C(x1, x2, Q, toeplitz1, toeplitz2, toeplitz3, autocor2, indices, k_size)
    #solve the Ca = -c using linear regression for each colour channel
    c = autocor2[x1, x2]
    a = np.linalg.lstsq(C.astype(np.float32), c.astype(np.float32))[0]
    return a

@jit(nopython=True)
def estimate_frame(I1, I2, A, k_width, k_off, b):
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0] ):
            for j in range(0, A.shape[1] ):
                kernel = A[i, j, :].reshape((2 * k_width,  k_width))
                k_sum = np.sum(kernel)
                if(k_sum != 0):
                   kernel = kernel/np.sum(kernel)
                patch1 = I1[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 ,channel] 
                patch2 = I2[i + b - k_off: i + b + k_off + 1, j + b - k_off: j + b + k_off + 1 ,channel]
                patch = np.concatenate((patch1, patch2), axis = 0)
                mask = kernel * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted

@jit(nopython=True)
def estimate_frame_motion(I1, I2, A, k_width, k_off, b, mvs1, mvs2):
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for channel in range(0, 3):
        for i in range(0, A.shape[0] ):
            for j in range(0, A.shape[1] ):
                kernel = A[i, j, :].reshape((2 * k_width,  k_width))
                k_sum = np.sum(kernel)
                if(k_sum != 0):
                   kernel = kernel/np.sum(kernel)
                y1 = i + mvs1[i + b, j + b, 0]
                x1 = j + mvs1[i + b, j + b, 1]
                y2 = i + mvs2[i + b, j + b, 0]
                x2 = j + mvs2[i + b, j + b, 1]
                patch1 = I1[y1 + b - k_off: y1 + b + k_off + 1, x1 + b - k_off: x1 + b + k_off + 1 ,channel] 
                patch2 = I2[y2 + b - k_off: y2 + b + k_off + 1, x2 + b - k_off: x2 + b + k_off + 1 ,channel]
                patch = np.concatenate((patch1, patch2), axis = 0)
                mask = kernel * patch
                predicted[i, j, channel] = np.sum(mask)
    return predicted

def estimate_coefficients(I, toeplitz1, toeplitz2, toeplitz3, autocor2, Q, q_d, k_width, b):  
    A = np.zeros((I.shape[0] - 2 * b, I.shape[1] - 2 * b, 2 * k_width ** 2))
    print("Estimating coefficients...")
    #calculate indices to be used
    k_size = k_width ** 2
    indices = np.empty(k_size * k_size)
    for i in range(0, k_size):
        for j in range(0, k_size):
            offset = Q[j] - Q[i]
            indices[i * k_size + j] = q_d[(offset[0], offset[1])]
        
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1] ):
            A[i, j] = calc_a(i + b, j + b, Q, toeplitz1, toeplitz2, toeplitz3, autocor2, indices, k_size)
    return A

def import_video():
    #read from config file
    input_video, start_frame, end_frame = read_input()
    #load video
    video = cv2.VideoCapture(input_video)
    #load initial frame
    ret, frame = video.read()

    if(ret == False):
        print("Video not found, exiting")
        return 0 
    y_lim = frame.shape[0]
    x_lim = frame.shape[1]
    print("Resolution is", y_lim, "x", x_lim)
    I = np.zeros((y_lim + 2*b, x_lim + 2*b, 3, end_frame - start_frame + 1))
    for i in range(0, start_frame):
        ret, frame = video.read()

    for i in range(0, end_frame - start_frame + 1):        
        I[ :, :, :, i] = cv2.copyMakeBorder(frame, b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
        ret, frame = video.read()
    return I

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

    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    
    Q = generate_Q(k_width)

    i_g = np.zeros((I1.shape[0], I1.shape[1], 3), dtype = np.int32)
    i_g[:, :, 0] = I1[:, :, 1]
    i_g[:, :, 1] = I2[:, :, 1]
    i_g[:, :, 2] = I3[:, :, 1]

    if(motion):
        mvs1 = motion_est(i_g[:, :, 0], i_g[:, :, 1], b, max_motion)
        mvs2 = motion_est(i_g[:, :, 2], i_g[:, :, 1], b, max_motion)
        c_g = get_c_m(i_g, Q, int(ac_block/2), mvs1, mvs2, b)
        C_g = get_C_m(i_g, Q, int(ac_block/2), mvs1, mvs2, b)
        A_g = estimate_coefficients_motion(c_g, C_g)
        predicted = estimate_frame_motion(I1, I3, A_g, k_width, k_off, b, mvs1, mvs2)
    else:
        double_Q = generate_Q(2 * k_width - 1)
        q_d = q_dict(double_Q)
        toeplitz1_g = generate_toeplitz(i_g[:, :, 0], i_g[:, :, 0], ac_block, k_width, double_Q, b)
        toeplitz2_g = generate_toeplitz(i_g[:, :, 2], i_g[:, :, 0], ac_block, k_width, double_Q, b)
        toeplitz3_g = generate_toeplitz(i_g[:, :, 2], i_g[:, :, 2], ac_block, k_width, double_Q, b)
        autocor_g  = generate_ac_tensor(i_g[:, :, 1], i_g[:, :, 0], i_g[:, :, 2], ac_block, k_width, Q, b)
        A_g = estimate_coefficients(I2[:, :, 1], toeplitz1_g, toeplitz2_g, toeplitz3_g, autocor_g, Q, q_d, k_width, b)
        predicted = estimate_frame(I1, I3, A_g, k_width, k_off, b)

    with open('../kernel_output/kernel_bi_2d.npy', 'wb') as k:
	    np.save(k, A_g)

    predicted[predicted > 255] = 255
    predicted[predicted < 0] = 0
    return predicted

#global variables

if(len(sys.argv) == 7):
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    image3 = sys.argv[3]
    out = sys.argv[4]
    k_width = int(sys.argv[5])
    interp_type = int(sys.argv[6])

elif (len(sys.argv) < 7 and len(sys.argv) >= 5):
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    image3 = sys.argv[3]
    out = sys.argv[4]
    k_width = 51
    interp_type = 0

else:
    image1 = '../images/image0.png'
    image2 = '../images/image1.png'
    image3 = '../images/image2.png'
    out = '../output/out.png'
    k_width = 3
    interp_type = 0
    ac_block = 11

def main():
    print("Kernel size: ", k_width)  
    print("Using 2D kernel and two frames...")  
    #load video frames as specified in config.txt
    print("Using 2D kernel...")
    motion = 0
    predicted = predict_frame(image1, image2, image3, k_width, ac_block, motion)
    print("PSNR is :", get_psnr(cv2.imread(image2), predicted))
    if(cv2.imwrite(out, predicted) == False):
        print("Error writing file!")
    else:
        print("Image written to file")

if __name__ == "__main__":
    main()