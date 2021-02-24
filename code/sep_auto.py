import numpy as np
from numba import jit, typed, types, cuda
import cv2
from tqdm import tqdm
import time

@cuda.jit
def cuda_mask(matrix, width, output, pad):
    y, x = cuda.grid(2)
    result = 0
    if(y + width >= matrix.shape[0] or x + width >= matrix.shape[1]):
        return
    for i in range(0, width):
        for j in range(0, width):
            result += matrix[y + i, x + j]
    output[y, x] = result


def mask(matrix, width):
    #find padding for matrix based on kernel width
    pad = int(width / 2)
    #add padding to border of matrix
    padded_matrix = np.zeros((matrix.shape[0] +  2 * pad, matrix.shape[1] + 2 * pad))
    padded_matrix[pad : -pad, pad: -pad] = matrix
    #threads per block
    TPB = (32, 32)
    #blocks per grid
    bpg_x = int(np.ceil(matrix.shape[0]/TPB[0]))
    bpg_y = int(np.ceil(matrix.shape[1]/TPB[1]))
    BPG = (bpg_x, bpg_y)
    #define output
    output = np.empty_like(matrix)
    cuda_mask[BPG, TPB](padded_matrix, width, output, pad)
    return output

def generate_ac_tensor(current_frame, last_frame, next_frame, ak_width, k_width, Q, b):
    ac_tensor = np.zeros((current_frame.shape[0], current_frame.shape[1], 2 * Q.shape[0]), dtype = np.int32)
    #print("Calculating second autocorrelation tensor...")
    for i in tqdm(range(0, Q.shape[0])):
        shifted = np.roll(last_frame, -Q[i, 0], axis = 0)
        shifted = np.roll(shifted, -Q[i, 1], axis = 1)
        result = np.multiply(current_frame, shifted)
        ac_tensor[:, :, i] = mask(result, ak_width)
        # ac_tensor[:, :, i] = shift_multiply(last_frame, current_frame, ak_width, Q[i])

    for i in tqdm(range(0, Q.shape[0])):
        shifted = np.roll(next_frame, -Q[i, 0], axis = 0)
        shifted = np.roll(shifted, -Q[i, 1], axis = 1)
        result = np.multiply(current_frame, shifted)
        ac_tensor[:, :, Q.shape[0] + i] = mask(result, ak_width)
        #ac_tensor[:, :, Q.shape[0] + i] = shift_multiply(next_frame, current_frame, ak_width, Q[i])
    return ac_tensor

def ac_tensor_uni(image1, image2, ak_width, k_width, Q, b):
    ac_tensor = np.zeros((image1.shape[0], image1.shape[1], Q.shape[0]), dtype = np.int32)
    #print("Calculating second autocorrelation tensor...")
    for i in tqdm(range(0, Q.shape[0])):
        shifted = np.roll(image1, -Q[i, 0], axis = 0)
        shifted = np.roll(shifted, -Q[i, 1], axis = 1)
        result = np.multiply(image2, shifted)
        ac_tensor[:, :, i] = mask(result, ak_width)
    return ac_tensor

def generate_toeplitz(image1, image2, ak_width, k_width, Q, b):
    ac_tensor = np.zeros((image1.shape[0], image1.shape[1], Q.shape[0]), dtype = np.int32)
    #print("Calculating Toeplitz tensor...")
    #print("Size is: ", size/10 ** 9, "Gb")
    for i in tqdm(range(0, Q.shape[0])):
        shifted = np.roll(image2, -Q[i, 0], axis = 0)
        shifted = np.roll(shifted, -Q[i, 1], axis = 1)
        result = np.multiply(image1, shifted)
        ac_tensor[:, :, i] = mask(result, ak_width)
    return ac_tensor

def q_dict(Q):
    qd = typed.Dict.empty(key_type = types.UniTuple(types.int64, 2), value_type=types.int64, )
    for i in range(Q.shape[0]):
        key = np.asarray([Q[i, 0], Q[i, 1]], dtype = np.int64)
        qd[tuple(key)] = i
    return qd

def generate_Q(width):
    Q = np.zeros((width, width, 2))
    #maximum offset
    off = int((width  - (width % 2))/2)
    #e.g for a 3x3 kernel centered at current pixel max offset is 
    for i in range(0, width):
        Q[i, :] = i - off
        Q[:, i] = i - off
    Q = Q.reshape((width * width, 2)).astype(np.int64)
    
    return Q

def main():
    return 0


if __name__ == "__main__":
    main()
