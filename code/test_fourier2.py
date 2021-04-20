import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from interp_2d import predict_frame as predict_frame2d
from interp_uni_2d import predict_frame_uni as predict_frame_uni2d
from interp import predict_frame
from new_motion_interp import predict_frame_uni
import math

def sign(out_dir):
    out_file = open(out_dir + '/sign.txt', 'w')
    out_file.write('test_fourier2.py')

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

    out_dir = '../graphs/ice/' + two_str + 'kk' + bd_string  + motion_str
    os.mkdir(out_dir)
    sign(out_dir)

    for i in range(2, len(image_names)):
        os.system("python ../sepconv-slomo/run.py --model lf --first " + path_names[i - 2] + " --second " + path_names[i] + " --out ../niklaus_out/out.png")
        niklaus = cv2.imread("../niklaus_out/out.png")
        original = cv2.imread(path_names[i - 1])
        niklaus_psnr[i - 2] = get_psnr(niklaus, original)
  
    for k in range(0, kernel_ac.shape[0]):
        for i in range(2, len(image_names)):
            print("k_width: ", kernel_ac[k, 0], "ac_block: ", kernel_ac[k, 1], "image: ", path_names[i - 1])
            interpolated = predict_frame_uni(path_names[i - 2], path_names[i], kernel_ac[k, 0], kernel_ac[k, 1]).astype(np.uint8)
            original = cv2.imread(path_names[i - 1])
            psnr[k, i - 2] = get_psnr(interpolated, original)
    
    plt.figure()
    x_axis = np.arange(psnr.shape[1])
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + motion_str)
    plt.ylabel("PSNR")
    plt.xlabel("Frame")

    for k in range(psnr.shape[0]):
        plt.plot(x_axis, psnr[k], 'x-', label = 'kernel = ' + str(kernel_ac[k, 0]) + ' block = ' + str(kernel_ac[k, 1]))
    plt.plot(x_axis, niklaus_psnr, 'x-', label = 'SNASC')
    plt.legend()
    plt.grid()
    plt.savefig(out_dir +'/graph' + motion_str + bd_string + '.png')
    plt.close()

def main():
    two_d = 1
    motion = 1
    bi_direct = 0
    kernel_ac = np.array([[3, 5], [5, 7], [7, 9]])
    test(kernel_ac, motion, bi_direct, two_d)

main()
if __name__ == 'main':
    main()
