import numpy as np
import matplotlib.pyplot as plt


matrix = np.random.rand(10, 10)
dots = np.random.rand(20, 2)

def gaussian_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

gaussian_fil = gaussian_filter(20, 3)
print(gaussian_fil)


plt.imshow(gaussian_fil)
plt.scatter(dots[:, 0], dots[:, 1], c='red', s=30)
plt.show()

