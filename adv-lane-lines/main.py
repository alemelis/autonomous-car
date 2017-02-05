import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def openImage():
    return mpimg.imread("imgs/signs_vehicles_xygrad.png")

def img2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def absSobelThresh(gray, orient='x', kernel=3, thresh=(0,255)):

    if orient == 'x':
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    else:
        gradient = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    abs_grad = np.absolute(gradient)

    scaled = np.uint8(255*abs_grad/np.max(abs_grad))

    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return binary

def magThresh(gray, kernel=3, thresh=(0,255)):

    g_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    g_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    magnitude = np.sqrt(g_x**2 + g_y**2)

    abs_mag = np.absolute(magnitude)

    scaled = np.uint8(255*abs_mag/np.max(abs_mag))

    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return binary

def dirThreshold(gray, kernel, thresh=(0, np.pi/2)):

    g_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    g_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    direction = np.arctan2(g_y, g_x)

    abs_dir = np.absolute(direction)

    binary = np.zeros_like(abs_dir)
    binary[(abs_dir >= thresh[0]) & (abs_dir <= thresh[1])] = 1

    return binary



img = openImage()
gray = img2gray(img)

ksize = 9

gradx = absSobelThresh(gray, 'x', ksize, (50, 200))
grady = absSobelThresh(gray, 'y', ksize, (50, 200))
mag_binary = magThresh(gray, ksize, (100, 200))
dir_binary = dirThreshold(gray, ksize, (np.pi/3, np.pi/2))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) |
         ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.imshow(combined, cmap='gray')
plt.show()
