import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from interp_2d import predict_frame as predict_frame2d
from interp_uni_2d import predict_frame_uni as predict_frame_uni2d
from interp import predict_frame
from interp_uni import predict_frame_uni
import math

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

def test(kernel_ac, motion, bi_direct, two_d):
    #parameters
    image_names = os.listdir('../images')
    path_names = ['../images/' + image for image in image_names]
    niklaus_psnr = np.zeros((len(image_names) - 2))
    
    psnr = np.zeros((kernel_ac.shape[0], len(image_names) - 2))


    for i in range(2, len(image_names)):
        os.system("python ../sepconv-slomo/run.py --model lf --first " + path_names[i - 2] + " --second " + path_names[i] + " --out ../niklaus_out/out.png")
        niklaus = cv2.imread("../niklaus_out/out.png")
        original = cv2.imread(path_names[i - 1])
        niklaus_psnr[i - 2] = get_psnr(niklaus, original)
  
    for k in range(0, kernel_ac.shape[0]):
        for i in range(2, len(image_names)):
            print("k_width: ", kernel_ac[k, 0], "ac_block: ", kernel_ac[k, 1], "image: ", path_names[i - 1])
            if(bi_direct):
                if(two_d):
                    interpolated = predict_frame2d(path_names[i - 2], path_names[i - 1], path_names[i], kernel_ac[k, 0], kernel_ac[k, 1], motion).astype(np.uint8)
                else:
                    interpolated = predict_frame(path_names[i - 2], path_names[i - 1], path_names[i], kernel_ac[k, 0], kernel_ac[k, 1], motion).astype(np.uint8)
            else:
                if(two_d):
                    interpolated = predict_frame_uni2d(path_names[i - 2], path_names[i - 1], kernel_ac[k, 0], kernel_ac[k, 1], motion).astype(np.uint8)
                else:
                    interpolated = predict_frame_uni(path_names[i - 2], path_names[i - 1], kernel_ac[k, 0], kernel_ac[k, 1], motion).astype(np.uint8)
            original = cv2.imread(path_names[i - 1])
            psnr[k, i - 2] = get_psnr(interpolated, original)

    if(motion):
        motion_str = 'motion'
    else:
        motion_str = ''
    if(bi_direct):
        bd_string = 'bi'
    else:
        bd_string = 'uni'
    if(two_d):
        two_str = '2d'
    else:
        two_str = 'plus'

    os.mkdir('../graphs/' + two_str + 'k' + bd_string  + motion_str)
    plt.figure()
    x_axis = np.arange(psnr.shape[1])
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + motion_str)
    plt.ylabel("PSNR")
    plt.xlabel("Frame")

    for k in range(psnr.shape[0]):
        plt.plot(x_axis, psnr[k], 'x-', label = 'kernel = ' + str(kernel_ac[k, 0]) + ' block = ' + str(kernel_ac[k, 1]))
    plt.plot(x_axis, niklaus_psnr, 'x-', label = 'niklaus')
    plt.legend()
    plt.savefig('../graphs/'  + two_str +'k' + bd_string + motion_str +'/graph' + motion_str + bd_string + '.png')
    plt.close()

def main():
    kernel_ac = np.array([[3,5], [5,7], [7,9]])
    two_d = 1
    bi_direct = 0
    motion = 1
    test(kernel_ac, motion, bi_direct, two_d)

    kernel_ac = np.array([[3,5], [5,5], [7,7]])
    two_d = 0
    bi_direct = 0
    motion = 1
    test(kernel_ac, motion, bi_direct, two_d)

    kernel_ac = np.array([[3,5], [5,7], [7,9]])
    two_d = 1
    bi_direct = 0
    motion = 0
    test(kernel_ac, motion, bi_direct, two_d)

    kernel_ac = np.array([[3,5], [5,5], [7,7]])
    two_d = 0
    bi_direct = 0
    motion = 0
    test(kernel_ac, motion, bi_direct, two_d)




main()
if __name__ == 'main':
    main()
