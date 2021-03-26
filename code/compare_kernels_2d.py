import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
from numpy import ma
from matplotlib import cbook
import math
import cv2
#load niklaus kernels


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


def main():
    k_width = 13
    k_off = int(k_width / 2)
    i = 75
    j = 75
    image1 = plt.imread('../images/image0.png')
    image2 = plt.imread('../images/image2.png')
    n_centre = int(51/2) - 1
    with open("../sepconv/horizontal.npy", 'rb') as h:
        prev_h = np.load(h)
        next_h = np.load(h)

    with open("../sepconv/vertical.npy", 'rb') as v:
        prev_v = np.load(v)
        next_v = np.load(v)

    with open('../kernel_output/kernel_bi.npy', 'rb') as a:
        A = np.load(a)
    print(prev_v.shape)
   
    kernel = A[i, j, :].reshape((2 * k_width,  k_width))
    k_sum = np.sum(kernel)
    if(k_sum != 0):
        kernel = kernel/np.sum(kernel)

    kernel_p = kernel[:k_width]
    kernel_f = kernel[k_width:]
    niklaus_previous = np.outer(prev_h[:, :, i + 100, j + 100], prev_v[:, :, i + 100, j + 100])
    niklaus_next = np.outer(next_h[:,:,i,j], next_v[:, :, i + 100, j + 100])
    print("Sum", np.sum(niklaus_previous - niklaus_next))
  
    plt.figure()
    plt.title("Niklaus previous - segment")
    plt.imshow(image1)
    #plt.imshow(image1[i - k_off : i + k_off + 1, j - k_off : j + k_off + 1])
    #plt.imshow(niklaus_previous[n_centre - k_off : n_centre + k_off, n_centre - k_off : n_centre + k_off], cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.figure()
    plt.title("Niklaus next - segment")
    plt.imshow(image2[i - k_off : i + k_off + 1, j - k_off : j + k_off + 1])
    plt.imshow(niklaus_next[n_centre - k_off : n_centre + k_off, n_centre - k_off : n_centre + k_off], cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.figure()
    plt.title("Niklaus previous - full")
    plt.imshow(image1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(niklaus_previous, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    
    plt.colorbar()

    plt.figure()
    plt.title("Niklaus next - full")
    plt.imshow(image2[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(niklaus_next, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.figure()
    plt.title("3DAR previous")
    plt.imshow(image1[i - k_off : i + k_off + 1, j - k_off : j + k_off + 1])
    plt.imshow(kernel_p, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.figure()
    plt.title("3DAR next")
    plt.imshow(image2[i - k_off : i + k_off + 1, j - k_off : j + k_off + 1])
    plt.imshow(kernel_f, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()