import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from interp_2d import predict_frame as predict_frame2d
from interp_uni_2d import predict_frame_uni as predict_frame_uni2d
from interp import predict_frame
from new_motion_interp import predict_frame_uni
import math

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

def sign(out_dir):
    out_file = open(out_dir + '/sign.txt', 'w')
    out_file.write('test_fourier.py')

def test(k_width, ac_block, motion, bi_direct, two_d):
    #parameters
    image_names = os.listdir('../images')
    path_names = ['../images/' + image for image in image_names]
    psnr = np.zeros((4, len(ac_block), len(image_names) - 2))
    niklaus_psnr = np.zeros((4, len(image_names) - 2))

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

    out_dir = '../graphs/ice/' + two_str + 'kk' + str(k_width) + bd_string  + motion_str
    os.mkdir(out_dir)
    sign(out_dir)

    for i in range(2, len(image_names)):
        os.system("python ../sepconv-slomo/run.py --model lf --first " + path_names[i - 2] + " --second " + path_names[i] + " --out ../niklaus_out/out2.png")

        niklaus = cv2.imread("../niklaus_out/out2.png")
        original = cv2.imread(path_names[i - 1])
        niklaus_psnr[0, i - 2] = get_psnr(niklaus, original)
        niklaus_psnr[1, i - 2] = get_psnr(niklaus[:, :, 0], original[:, :, 0])
        niklaus_psnr[2, i - 2] = get_psnr(niklaus[:, :, 1], original[:, :, 1])
        niklaus_psnr[3, i - 2] = get_psnr(niklaus[:, :, 2], original[:, :, 2])

    for k in range(0, len(ac_block)):
        for i in range(2, len(image_names)):
            print("k_width: ", k_width, "ac_block: ", ac_block[k], "image: ", path_names[i - 1])
            interpolated = predict_frame_uni(path_names[i - 2], path_names[i], k_width, ac_block[k]).astype(np.uint8)
            original = cv2.imread(path_names[i - 1])
            psnr[0, k, i - 2] = get_psnr(interpolated, original)
            psnr[1, k, i - 2] = get_psnr(interpolated[:, :, 0], original[:, :, 0])
            psnr[2, k, i - 2] = get_psnr(interpolated[:, :, 1], original[:, :, 1])
            psnr[3, k, i - 2] = get_psnr(interpolated[:, :, 2], original[:, :, 2])

    
    plt.figure()
    x_axis = np.arange(psnr.shape[2])
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + str(k_width) + motion_str)
    plt.ylabel("PSNR")
    plt.xlabel("Frame")
    print(x_axis.shape)
    print(psnr[0, 0].shape)
    for k in range(psnr.shape[1]):
        plt.plot(x_axis, psnr[0, k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.plot(x_axis, niklaus_psnr[0], 'x-', label = 'niklaus')
    plt.legend()
    plt.grid()
    plt.savefig(out_dir +'/graph' + str(k_width) + motion_str + bd_string + '.png')

    plt.figure()
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + str(k_width) + motion_str)
    plt.ylabel("PSNR (red channel only)")
    plt.xlabel("Frame")
    for k in range(psnr.shape[1]):
        plt.plot(x_axis, psnr[1, k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.plot(x_axis, niklaus_psnr[1], 'x-', label = 'niklaus')
    plt.legend()
    plt.grid()
    plt.savefig(out_dir + '/graph_red'+ str(k_width) + motion_str + bd_string + '.png')

    plt.figure()
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + str(k_width) + motion_str)
    plt.ylabel("PSNR (green channel only)")
    plt.xlabel("Frame")
    for k in range(psnr.shape[1]):
        plt.plot(x_axis, psnr[2, k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.plot(x_axis, niklaus_psnr[2], 'x-', label = 'niklaus')
    plt.legend()
    plt.grid()
    plt.savefig(out_dir +'/graph_green' + str(k_width) + motion_str + bd_string + '.png')
    plt.figure()

    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes " + str(k_width) + motion_str)
    plt.ylabel("PSNR (blue channel only)")
    plt.xlabel("Frame")
    for k in range(psnr.shape[1]):
        plt.plot(x_axis, psnr[3, k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.plot(x_axis, niklaus_psnr[3], 'x-', label = 'niklaus')
    plt.legend()
    plt.grid()
    plt.savefig(out_dir + '/graph_blue' + str(k_width) + motion_str + bd_string + '.png')
    plt.close()

def main():

    ##
    two_d = 1
    bi_direct = 0
    motion = 1

    k_width = 3
    ac_block = [5, 7, 9, 11, 13]
    test(k_width, ac_block, motion, bi_direct, two_d)

    k_width = 5
    ac_block = [7, 9, 11, 13, 15]
    test(k_width, ac_block, motion, bi_direct, two_d)
    ##
    k_width = 7
    ac_block = [9, 11, 13, 15, 17]
    test(k_width, ac_block, motion, bi_direct, two_d)


    


main()
if __name__ == 'main':
    main()
