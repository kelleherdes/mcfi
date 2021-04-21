import numpy as np
from numba import jit, cuda
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

@jit(nopython=True)
def convolve2d(image, N):
    pad = int(N / 2)
    padded_image = np.zeros((image.shape[0] + 2 * pad, image.shape[1] + 2 * pad))
    padded_image[pad : - pad, pad : -pad] = image
    product = np.empty_like(padded_image)
   
    for i in range(pad, pad + image.shape[0]):
        for j in range(pad, pad + image.shape[1]):
            product[i, j] = np.sum(padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1])
    return product[pad : -pad, pad : -pad]

@jit(nopython=True)
def convolve(vector, N):
    pad = int(N/2)
    padded_vector = np.zeros(vector.shape[1] + 2 * pad)
    padded_vector[pad:-pad] = np.abs(vector)
    output = np.zeros(vector.shape[1])
    for i in range(0, vector.shape[1]):
        output[i] = np.sum(padded_vector[i: i + N])
    return output

@jit(nopython=True)
def compress_kernels(p_h, n_h, p_v, n_v, N):
    off = int(N / 2)
    compressed = np.zeros((p_h.shape[2], p_h.shape[3], 2, 2, N))
    vectors = np.zeros((p_h.shape[2], p_h.shape[3], 2, 2))
    for i in range(0, p_h.shape[2]):
        for j in range(0, p_h.shape[3]):
            ph_max = convolve(p_h[:, :, i, j], N)
            nh_max = convolve(n_h[:, :, i, j], N)
            pv_max = convolve(p_v[:, :, i, j], N)
            nv_max = convolve(n_v[:, :, i, j], N)
            
            max_p = np.array((np.argmax(pv_max), np.argmax(ph_max)))
            max_n = np.array((np.argmax(nv_max), np.argmax(nh_max))) 

            compressed[i, j, 0, 0] = p_v[ :, max_p[0] - off : max_p[0] + off + 1, i, j]
            compressed[i, j, 0, 1] = p_h[ :, max_p[1] - off : max_p[1] + off + 1, i, j]
            compressed[i, j, 1, 0] = n_v[ :, max_n[0] - off : max_n[0] + off + 1, i, j]
            compressed[i, j, 1, 1] = n_h[ :, max_n[1] - off : max_n[1] + off + 1, i, j]
            max_p -= 25
            max_n -= 25
            vectors[i, j, 0] = max_p
            vectors[i, j, 1] = max_n
    return vectors, compressed

@jit(nopython=True)
def apply_kernels(i1, i2, p_h, n_h, p_v, n_v, b):
    interp = np.zeros((i1.shape[0] - 2 * b, i1.shape[1] - 2 * b, 3))
    off = int(51 / 2)
    for i in range(b, i1.shape[0] - b):
        for j in range(b, i1.shape[1] - b):
            kernel_p = np.outer(p_v[:, :, i - b, j - b], p_h[:, :, i - b, j - b])
            kernel_n = np.outer(n_v[:, :, i - b, j - b], n_h[:, :, i - b, j - b])
            interp[i - b, j - b, 0] = np.sum(i1[i - off : i + off + 1, j - off : j + off + 1, 0] * kernel_p + i2[i - off : i + off + 1, j - off : j + off + 1, 0] * kernel_n)
            interp[i - b, j - b, 1] = np.sum(i1[i - off : i + off + 1, j - off : j + off + 1, 1] * kernel_p + i2[i - off : i + off + 1, j - off : j + off + 1, 1] * kernel_n)
            interp[i - b, j - b, 2] = np.sum(i1[i - off : i + off + 1, j - off : j + off + 1, 2] * kernel_p + i2[i - off : i + off + 1, j - off : j + off + 1, 2] * kernel_n)
    return interp.astype(np.uint8)

@jit(nopython=True)
def apply_compressed_kernels(i1, i2, compressed, vectors, b, N):
    interp = np.zeros((i1.shape[0] - 2 * b, i1.shape[1] - 2 * b, 3))
    off = int(N / 2)
    for i in range(b, i1.shape[0] - b):
        for j in range(b, i1.shape[1] - b):
            kernel_p = np.outer(compressed[i - b, j - b, 0, 0], compressed[i - b, j - b, 0, 1])
            kernel_n = np.outer(compressed[i - b, j - b, 1, 0], compressed[i - b, j - b, 1, 1])
            k_sum = np.sum(kernel_p + kernel_n)
            kernel_p /= k_sum
            kernel_n /= k_sum
            y1 = i + vectors[i - b, j - b, 0, 0]
            x1 = j + vectors[i - b, j - b, 0, 1]
            y2 = i + vectors[i - b, j - b, 1, 0]
            x2 = j + vectors[i - b, j - b, 1, 1]
            interp[i - b, j - b, 0] = np.sum(i1[y1 - off : y1 + off + 1, x1 - off : x1 + off + 1, 0] * kernel_p + i2[y2 - off : y2 + off + 1, x2 - off : x2 + off + 1, 0] * kernel_n)
            interp[i - b, j - b, 1] = np.sum(i1[y1 - off : y1 + off + 1, x1 - off : x1 + off + 1, 1] * kernel_p + i2[y2 - off : y2 + off + 1, x2 - off : x2 + off + 1, 1] * kernel_n)
            interp[i - b, j - b, 2] = np.sum(i1[y1 - off : y1 + off + 1, x1 - off : x1 + off + 1, 2] * kernel_p + i2[y2 - off : y2 + off + 1, x2 - off : x2 + off + 1, 2] * kernel_n)
    return interp.astype(np.uint8)

def main():
    with open("../sepconv/fb_0_h.npy", 'rb') as h:
        prev_h = np.copy(np.load(h)[:, :, :, :])

    with open("../sepconv/fb_2_h.npy", 'rb') as h:   
        next_h = np.copy(np.load(h)[:, :, :, :])

    with open("../sepconv/fb_0_v.npy", 'rb') as v:
        prev_v = np.copy(np.load(v)[:, :, :, :])

    with open("../sepconv/fb_2_v.npy", 'rb') as v:  
        next_v = np.copy(np.load(v)[:, :, :, :])

    b = 32
    image1 = cv2.copyMakeBorder(cv2.imread('../football/image0.png'), b, b, b, b, cv2.BORDER_REFLECT)
    image2 = cv2.imread('../football/image1.png')
    image3 = cv2.copyMakeBorder(cv2.imread('../football/image2.png'), b, b, b, b, cv2.BORDER_REFLECT)
    image_out = cv2.imread("../niklaus_results/fb_1.png")
    N = 5
    v, c = compress_kernels(prev_h, next_h, prev_v, next_v, N)
    interp_c = apply_compressed_kernels(image1, image3, c, v, b, N)
    psnr_c = get_psnr(interp_c, image2)
    interp = apply_kernels(image1, image3, prev_h, next_h, prev_v, next_v, b)
    psnr = get_psnr(interp, image2)
    psnr_dnn = get_psnr(image_out, image2)
    print("PSNR", psnr)
    print("PSNR dnn", psnr_dnn)
    print("Compressed psnr", psnr_c)

    cv2.imshow('interp_c', interp_c)
    cv2.imshow('interp', interp)
    cv2.imshow('DNN', image_out)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()