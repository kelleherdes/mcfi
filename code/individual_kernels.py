import numpy as np
import cv2
import csv
import os 
import sys
from numba import jit, typed, types
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
from numpy import ma
from tqdm import tqdm
from matplotlib import cbook
from scipy import signal
from motion import motion_est
import math

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

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (value - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


def distribution(kernel, name):
    rounded = np.round(kernel, 2)
    low = np.min(rounded)
    high = np.max(rounded)
    x = np.arange(low, high + 0.01, 0.01).astype(np.float16)
    pmf = np.zeros(x.shape)
    unique, counts = np.unique(rounded, return_counts=True)
    for i in range(0, unique.shape[0]):
        pmf[x == unique[i]] = counts[i]
    pmf = pmf/2601
    plt.figure()
    plt.title("Probability density function for " + name)
    plt.plot(x, pmf)
    
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

def get_sign_error_reduced(niklaus, a):
    a1 = np.copy(niklaus)
    a1[abs(a1) < 0.001] = 0
    a2 = np.copy(a)
    N = np.count_nonzero(a1)

    a2[a1 == 0] = 0
    a1[a1 > 0] = 1
    a1[a1 < 0] = 0
    a2[a2 > 0] = 1
    a2[a2 < 0] = 0
    diff = (a1 - a2) ** 2
    return 100 * np.sum(diff)/N

def get_sign_error(a1, a2):
    a1_sign = np.copy(a1)
    a1_sign[a1_sign > 0] = 1
    a1_sign[a1_sign < 0] = 0
    a2_sign = np.copy(a2)
    a2_sign[a2_sign < 0] = 0
    a2_sign[a2_sign > 0] =  1
    diff = (a1_sign - a2_sign) ** 2
    return 100 * np.sum(diff)/(a1.shape[0] * a1.shape[1])

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
def estimate_frame(I1, A, k_width, b):
    k_off = int(k_width / 2)
    #p is past, f is future
    predicted = np.zeros((A.shape[0], A.shape[1], 3))
    y = 0
    x = 0
    #for each colour channel, use the corresponding AR coefficients to estimate future frame 
    for c in range(0, 3):
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[1]):
                kernel_p                   = A[i, j].reshape((k_width, k_width))
                k_sum = np.sum(kernel_p)
                if(k_sum != 0):
                   kernel_p = kernel_p/k_sum
                y = i + b
                x = j + b
                patch = I1[y - k_off: y + k_off + 1, x - k_off: x + k_off + 1, c] 
                mask = kernel_p * patch
                predicted[i, j, c] = np.sum(mask)
    return predicted

def estimate_coefficients(image1, image2, Q, k_width, ac_block, b):  
    A = np.zeros((image1.shape[0] - 2 * b, image2.shape[1] - 2 * b, k_width * k_width))
    print("Estimating coefficients...")
    for i in tqdm(range(0, A.shape[0])):
        for j in range(0, A.shape[1]):
            A[i, j] = simple_kernel(image1, image2, np.array([i + b, j + b]).astype(np.int64), k_width, ac_block, Q)
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
        k_width = 51
        ac_block = 21

    print("Kernel size:", k_width)
    print("Using 2D kernel and one frame...")
    #kernel max offsets (the max index to be used)    
    b = ac_block

    with open("../sepconv/horizontal.npy", 'rb') as h:
        prev_h = np.load(h)
        next_h = np.load(h)

    with open("../sepconv/vertical.npy", 'rb') as v:
        prev_v = np.load(v)
        next_v = np.load(v)

    
    plt.imshow(plt.imread(image1))
    point = np.asarray(plt.ginput(1, timeout = 0))
    plt.close()
    i = int(point[0, 1])
    j = int(point[0, 0])

    print(i, j)
    cen = int(k_width / 2)

    niklaus_previous = np.outer(prev_h[ :, :, i, j], prev_v[:, :, i, j])
    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)

    Q = generate_Q(k_width)
    a = simple_kernel(I1, I2, np.array([i + b, j + b]), k_width, ac_block, Q).reshape((k_width, k_width))

    I1 = plt.imread(image1)
    I2 = plt.imread(image2)
    plt.figure()
    plt.title("Niklaus ")
    plt.imshow(I1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(niklaus_previous, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.figure()
    plt.title("3DAR ")
    plt.imshow(I1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(a, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()
    a = a/2 * np.sum(a)
    #/////////////////////////////
    #/////error calculation//////
    #///////////////////////////
    #mse
    n1 = np.copy(niklaus_previous)
    a1 =  np.copy(a)
    n1[abs(n1) < 0.001] = 0
    a1[n1 == 0] = 0
    N = np.count_nonzero(n1)
    print("Nonzero points: ", N)
    mse = get_mse(niklaus_previous, a)
    print("MSE is: ", mse)
    #reduced mse
    mse_reduced = get_mse_reduced(n1, a1)
    print("Reduced MSE is: ", mse_reduced)
    #sign error
    sign_error = get_sign_error(a, niklaus_previous)
    print("Sign error is: ", sign_error)
    reduced_sign_error = get_sign_error_reduced(niklaus_previous, a)
    print("Reduced sign error is: ", reduced_sign_error)
    print("Niklaus centre: ", niklaus_previous[cen, cen])
    print("3DAR centre: ", a[cen, cen])
    print("Niklaus mean: ", np.mean(niklaus_previous))
    print("Niklaus standard deviation: ", np.std(niklaus_previous))
    print("Niklaus absolute sum: ", np.sum(abs(niklaus_previous)))
    print("3DAR mean: ", np.mean(a))
    print("3DAR standard deviation: ", np.std(a))
    print("3DAR absolute sum", np.sum(abs(a)))

    a_flatten = a.flatten()
    n_flatten = niklaus_previous.flatten()
    distribution(a_flatten, '3DAR')
    distribution(n_flatten, 'Niklaus')

    #/////////////////////////////
    #///////////graphs///////////
    #///////////////////////////
    # plt.figure()
    # plt.title("3DAR previous reduced")
    # plt.imshow(I1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    # plt.imshow(a1, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    # plt.colorbar()

    # plt.figure()
    # plt.title("Niklaus previous reduced")
    # plt.imshow(I1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    # plt.imshow(n1, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    # plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()