import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
from numpy import ma
from matplotlib import cbook
import cv2
from motion import motion_est
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
    i = 100
    j = 100
    motion = 1
    n_centre = int(51/2) - 1
    n_off = int(51/2)
    b = n_off - k_off
    image1 = plt.imread('../images/image0.png')
    image2 = plt.imread('../images/image1.png')
    image3 = plt.imread('../images/image2.png')
    mvs1 = motion_est(cv2.copyMakeBorder(image1[:,:,1], 10, 10, 10, 10, cv2.BORDER_CONSTANT), cv2.copyMakeBorder(image2[:,:,1], 10, 10, 10, 10, cv2.BORDER_CONSTANT), 10, 10)
    mvs2 = motion_est(cv2.copyMakeBorder(image3[:,:,1], 10, 10, 10, 10, cv2.BORDER_CONSTANT), cv2.copyMakeBorder(image2[:,:,1], 10, 10, 10, 10, cv2.BORDER_CONSTANT), 10, 10)
    plt.imshow(image1)
    point = np.asarray(plt.ginput(1, timeout = 0))
    plt.close()
    i = int(point[0, 1])
    j = int(point[0, 0])

    #for quiver
    Y = np.arange(51)
    X = np.arange(51)
    X, Y = np.meshgrid(Y, X)
    U1 = -mvs1[i - n_off: i + n_off + 1, j - n_off: j + n_off + 1, 1]
    V1 =  mvs1[i - n_off: i + n_off + 1, j - n_off: j + n_off + 1, 0]
    print(mvs1[i, j])

    U2 = -mvs2[i - n_off: i + n_off + 1, j - n_off: j + n_off + 1, 1]
    V2 = mvs2[i - n_off: i + n_off + 1, j - n_off: j + n_off + 1, 0]

    with open("../sepconv/horizontal.npy", 'rb') as h:
        prev_h = np.load(h)
        next_h = np.load(h)

    with open("../sepconv/vertical.npy", 'rb') as v:
        prev_v = np.load(v)
        next_v = np.load(v)

    with open('../kernel_output/kernel_uni1.npy', 'rb') as a:
        A1 = np.load(a)

    with open('../kernel_output/kernel_uni2.npy', 'rb') as a:
        A2 = np.load(a)
    
    kernel_p = A1[i, j].reshape((k_width, k_width))
    kernel_f = A2[i, j].reshape((k_width, k_width))
    kernel_p /= np.sum(kernel_p)
    kernel_f /= np.sum(kernel_f)
    niklaus_previous = np.outer(prev_h[ :, :, i, j], prev_v[:, :, i, j])
    niklaus_next = np.outer(next_h[ :, :, i, j], next_v[ :, :, i, j])

    kernel_p = cv2.copyMakeBorder(kernel_p, b, b, b, b, cv2.BORDER_CONSTANT)
    kernel_f = cv2.copyMakeBorder(kernel_f, b, b, b, b, cv2.BORDER_CONSTANT)
    
    kernel_p = np.roll(kernel_p, -mvs1[i, j, 0], axis = 0)
    kernel_p = np.roll(kernel_p, -mvs1[i, j, 1], axis = 1)

    kernel_f = np.roll(kernel_f, -mvs2[i, j, 0], axis = 0)
    kernel_f = np.roll(kernel_f, -mvs2[i, j, 1], axis = 1)

    skip=(slice(None, None, 2),slice(None,None,2))
    
    plt.figure()
    plt.title("Niklaus previous - full")
    plt.imshow(image1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(niklaus_previous, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()
    plt.quiver(X[skip], Y[skip], U1[skip], V1[skip], color = 'g', scale = 51)

    plt.figure()
    plt.title("Niklaus next - full")
    plt.imshow(image3[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(niklaus_next, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()
    plt.quiver(X[skip], Y[skip], U2[skip], V2[skip], color = 'g', scale = 51)

    plt.figure()
    plt.title("3DAR previous")
    plt.imshow(image1[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(kernel_p, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()
    plt.quiver(X[skip], Y[skip], U2[skip], V2[skip], color = 'g', scale = 51)

    plt.figure()
    plt.title("3DAR next")
    plt.imshow(image3[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.imshow(kernel_f, cmap='seismic', interpolation='nearest', norm=MidPointNorm(midpoint=0), alpha=0.5)
    plt.colorbar()
    plt.quiver(X[skip], Y[skip], U1[skip], V1[skip], color = 'g', scale = 51)

    plt.figure()
    plt.title("Ground truth")
    plt.imshow(image2[i - int(51/2) : i + int(51/2) + 1, j - int(51/2) : j + int(51/2) + 1])
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    main()