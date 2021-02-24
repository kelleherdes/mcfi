import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from interp import predict_frame
import math

def get_psnr(image1, image2):
    mse = np.sum((image1 - image2) ** 2)/(image1.shape[0] * image2.shape[1])
    psnr = 10 * math.log10(255 ** 2/mse)
    return psnr

def main():
    #parameters
    k_width = 21
    ac_block = [11, 15, 19, 23, 27]
    #ac_block = [11]
    image_names = os.listdir('../images')
    path_names = ['../images/' + image for image in image_names]
    psnr = np.zeros((len(ac_block), len(image_names) - 2))

    blue_psnr =np.zeros((len(ac_block), len(image_names) - 2)) 
    for k in range(0, len(ac_block)):
        for i in range(2, len(image_names)):
            print("k_width: ", k_width, "ac_block: ", ac_block[k], "image: ", path_names[i - 1])
            interpolated = predict_frame(path_names[i - 2], path_names[i - 1], path_names[i], k_width, ac_block[k]).astype(np.uint8)
            original = cv2.imread(path_names[i - 1])
            psnr[k, i - 2] = get_psnr(interpolated, original)
            blue_psnr[k, i - 2] = get_psnr(interpolated[:,:,1], original[:,:,1])

        print("PSNR", psnr[k])
        print("Blue PSNR", blue_psnr[k])

    plt.figure()
    x_axis = np.arange(psnr.shape[1])
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes")
    plt.ylabel("PSNR")
    plt.xlabel("Frame")
    for k in range(psnr.shape[0]):
        print(psnr[k])
        plt.plot(x_axis, psnr[k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.legend()
    plt.savefig('../graphs/graph.png')

    plt.figure()
    x_axis = np.arange(psnr.shape[1])
    plt.title("Performance of 3DAR interpolation with various autocorrelation \n block sizes")
    plt.ylabel("PSNR (green channel)")
    plt.xlabel("Frame")
    for k in range(psnr.shape[0]):
        print(blue_psnr[k])
        plt.plot(x_axis, blue_psnr[k, :], 'x-', label = 'block = ' + str(ac_block[k]))
    plt.legend()
    plt.savefig('../graphs/graph_blue.png')
    plt.show()


main()
if __name__ == 'main':
    main()
