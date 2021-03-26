import numpy as np
import cv2


def main():
    #load input images
    image0 = cv2.imread("../images/image0.png")
    image1 = cv2.imread("../images/image1.png")

    #load niklaus kernels
    with open("../sepconv/horizontal.npy", 'rb') as h:
        h_prev = np.load(h)
        h_next = np.load(h)

    with open("../sepconv/vertical.npy", 'rb') as v:
        v_prev = np.load(v)
        v_next = np.load(v)
    print(h_prev.shape)

    
if __name__ == "__main__":
    main()
