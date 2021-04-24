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
from matplotlib import cbook, cm
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

def colour_array(a, mmin, mmax):
    r_a = np.copy(a)
    r_a[r_a < 0] = 0
    r_a = ((255/mmax) * r_a).astype(np.uint8)
    b_a = np.copy(a)
    b_a[b_a > 0] = 0
    b_a = abs(b_a)
    b_a = ((255/mmax) * b_a).astype(np.uint8)
    c_a = np.zeros((a.shape[0], a.shape[1], 3))
    c_a[:,:,0] = r_a
    c_a[:,:,2] = b_a 
    c_a = c_a
    return c_a.astype(np.uint8)

def distribution(kernel, name):
    rounded = np.round(kernel.flatten(), 2)
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
    plt.savefig(dir_out + name + '_dist.png')
    
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

def get_mse_reduced(n, a):
    n1 = np.copy(n)
    a1 =  np.copy(a)
    n1[abs(n1) < 0.001] = 0
    a1[n1 == 0] = 0
    N = np.count_nonzero(n1)
    print("Nonzero points: ", N, file = o_file)
    mse = np.sum((n1 - a1) ** 2)/N
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

def graph(name, image, kernel, mid, i, j, X, Y, U, V, n_off):
    c = int(51/2)
    skip=(slice(None, None, 4),slice(None,None,4))
    plt.figure()
    plt.title(name)
    plt.imshow(image[i - n_off : i + n_off + 1, j - n_off : j + n_off + 1])
    plt.imshow(kernel[c - n_off : c + n_off + 1, c - n_off : c + n_off + 1], cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=mid), alpha=0.5)
    plt.colorbar()
    plt.quiver(X[skip], Y[skip], U[skip], V[skip], color = 'm', angles='xy', scale_units='xy', scale=1)
    plt.savefig(dir_out  + name + '.png')


def graph_truth(name, image, i, j):
    plt.figure()
    plt.title(name)
    plt.imshow(image[i - 5 : i + 6, j - 5 : j + 6])
    plt.savefig(dir_out  + name + '.png')

def graph3d(name, kernel):
    plt.figure()
    x = y = np.arange(51)
    X, Y = np.meshgrid(x, y)
    plt.suptitle(name)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, kernel, cmap = cm.CMRmap)

def get_stats(n, a, name):
    cen = int(51 / 2)
    print("Metrics for " + name, file = o_file)
    mse = get_mse(n, a)
    print("MSE is: ", mse, file = o_file)
    #reduced mse
    mse_reduced = get_mse_reduced(n, a)
    print("Reduced MSE is: ", mse_reduced, file = o_file)
    sign_error = get_sign_error(n, a)
    print("Sign error is: ", sign_error, file = o_file)
    reduced_sign_error = get_sign_error_reduced(n, a)
    print("Reduced sign error is: ", reduced_sign_error, file = o_file)
    print("Niklaus centre: ", n[cen, cen], file = o_file)
    print("3DAR centre: ", a[cen, cen], file = o_file)
    print("3DAR max: ", np.max(a), file=o_file)
    print("SNASC max: ", np.max(n), file=o_file)
    print("Niklaus mean: ", np.mean(n), file = o_file)
    print("Niklaus standard deviation: ", np.std(n), file = o_file)
    print("Niklaus absolute sum: ", np.sum(abs(n)), file = o_file)
    print("3DAR mean: ", np.mean(a), file = o_file)
    print("3DAR standard deviation: ", np.std(a), file = o_file)
    print("3DAR absolute sum", np.sum(abs(a)), file = o_file)
    
    n_max = np.array(np.unravel_index(np.argmax(n, axis=None), n.shape))
    a_max = np.array(np.unravel_index(np.argmax(a, axis=None), a.shape))
    dist = n_max - a_max
    d_mag = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
    print("Distance between maximum points: ", d_mag, file = o_file)
    print("#######################", file = o_file)

def get_errors(i1, i2, i3, n_p, n_n, a_p, a_n, ab, i, j, b):
    off = int(51 / 2)
    truth = i2[i + b, j + b, 1]
    uni_prev = np.sum(i1[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * a_p)
    uni_next = np.sum(i3[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * a_n)
    bi = np.sum(i1[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * ab[:51] + i3[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * ab[51:])
    snasc = np.sum(i1[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * n_p + i3[i + b - off : i + b + off + 1, j + b - off : j + b + off + 1, 1] * n_n)
    print("Uni prev error", truth - uni_prev, file = o_file)
    print("Uni next error", truth - uni_next, file = o_file)
    print("Bi error", truth - bi, file = o_file)
    print("SNASC error", truth - snasc, file = o_file)

dir_out = r'C:\Users\deske\OneDrive\Documents\College\MAI\new\kernel_tests\football\motion\test3\\'
o_file = open(dir_out + 'console_output.txt', 'w')
#global variables
def main():
    if(len(sys.argv) == 7):
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        out = sys.argv[3]
        k_width = int(sys.argv[4])
        ac_block = int(sys.argv[5])
    
    else:
        image1 = '../football/image0.png'
        image2 = '../football/image1.png'
        image3 = '../football/image2.png'
        out = '../output/simple_out.png'
        k_width = 51
        ac_block = 53
        
    print("Kernel size:", k_width, file=o_file)
    print("Using 2D kernel and one frame...")
    #kernel max offsets (the max index to be used)    
    b = ac_block
    with open("../sepconv/fb_0_h.npy", 'rb') as h:
        prev_h = np.copy(np.load(h)[:, :, :, :])

    with open("../sepconv/fb_2_h.npy", 'rb') as h:   
        next_h = np.copy(np.load(h)[:, :, :, :])

    with open("../sepconv/fb_0_v.npy", 'rb') as v:
        prev_v = np.copy(np.load(v)[:, :, :, :])

    with open("../sepconv/fb_2_v.npy", 'rb') as v:  
        next_v = np.copy(np.load(v)[:, :, :, :])
    

    plt.imshow(plt.imread(image1))
    point = np.asarray(plt.ginput(1, timeout = 0))
    plt.close()
    i = int(point[0, 1])
    j = int(point[0, 0])
    
   
    print("Points are: ", i, j, file=o_file)
    n_off = 10

    n_p = np.outer(prev_v[ :, :, i, j], prev_h[:, :, i, j])
    n_n = np.outer(next_v[ :, :, i, j], next_h[:, :, i, j])
    I1 = cv2.copyMakeBorder(cv2.imread(image1), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I2 = cv2.copyMakeBorder(cv2.imread(image2), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    I3 = cv2.copyMakeBorder(cv2.imread(image3), b, b, b, b, cv2.BORDER_REFLECT).astype(np.int32)
    mvs1 = motion_est(I1[:, :, 1], I2[:, :, 1], b, b)
    mvs2 = motion_est(I3[:, :, 1], I2[:, :, 1], b, b)

    Y = np.arange(21)
    X = np.arange(21)
    X, Y = np.meshgrid(Y, X)
    U1 = -mvs1[i - n_off + b: i + n_off + 1 + b, j - n_off + b: j + n_off + 1 + b, 1]
    V1 =  mvs1[i - n_off + b: i + n_off + 1 + b, j - n_off + b: j + n_off + 1 + b, 0]

    U2 = -mvs2[i - n_off + b: i + n_off + 1 + b, j - n_off + b: j + n_off + 1 + b, 1]
    V2 =  mvs2[i - n_off + b: i + n_off + 1 + b, j - n_off + b: j + n_off + 1 + b, 0]

    Q = generate_Q(k_width)
    a_p = simple_kernel(I1[:,:,1], I2[:,:,1], np.array([i + b, j + b]), k_width, ac_block, Q).reshape((k_width, k_width))
    a_n = simple_kernel(I3[:,:,1], I2[:,:,1], np.array([i + b, j + b]), k_width, ac_block, Q).reshape((k_width, k_width))
    ab = simple_kernel2(I1[:,:,1], I2[:,:,1], I3[:,:,1], np.array([i + b, j + b]), k_width, ac_block, Q).reshape((2 * k_width, k_width))

    ab /= np.sum(ab)
    a_p /= np.sum(a_p)
    a_n /= np.sum(a_n)
    get_errors(I1, I2, I3, n_p, n_n, a_p, a_n, ab, i, j, b)

    a_p = a_p/(2 * np.sum(n_p))
    a_n = a_n/(2 * np.sum(n_n))
    ab_p = ab[:k_width]
    ab_n = ab[k_width:]
    get_stats(n_p, a_p, 'Unilateral previous')
    get_stats(n_n, a_n, 'Unilateral next')
    get_stats(n_p, ab_p, 'Bilateral previous')
    get_stats(n_n, ab_n, 'Bilateral next')
    distribution(ab_n, 'Bilateral Next')
    distribution(ab_p, 'Bilateral Previous')
    distribution(a_p, 'Unilateral Previous')
    distribution(a_n, 'Unilateral Next')
    distribution(n_n, 'SNASC Next')
    distribution(n_p, 'SNASC Previous')

    I1 = plt.imread(image1)
    I2 = plt.imread(image2)
    I3 = plt.imread(image3)
    #def graph(name,image, kernel, mid):
    graph_truth('Truth patch',       I2, i, j)
    graph('SNASC Previous',        I1, n_p, 0, i, j, X, Y, U1, V1, n_off)
    graph('SNASC Next',            I3, n_n, 0, i, j, X, Y, U2, V2, n_off)
    graph('3DAR Unilateral Previous',I1, a_p, 0, i, j, X, Y, U1, V1, n_off)
    graph('3DAR Unilateral Next',    I3, a_n, 0, i, j, X, Y, U2, V2, n_off)
    graph('3DAR Bilateral Previous', I1, ab_p, 0, i, j, X, Y, U1, V1, n_off)
    graph('3DAR Bilateral Next',     I3, ab_n, 0, i, j, X, Y, U2, V2, n_off)

    #plt.show()

if __name__ == "__main__":
    main()