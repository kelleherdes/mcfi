import numpy as np
import matplotlib.pyplot as plt

psnr = np.random.randint(10, size = (4, 5, 8))
x_axis = np.arange(psnr.shape[2])


plt.figure()
for k in range(psnr.shape[1]):
    plt.plot(x_axis, psnr[0, k])


plt.show()
    