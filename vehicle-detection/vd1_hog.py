import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import glob, time
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

def readImage(image_name):
    return mpimg.imread(image_name)

def cvtImage(img, color_space='GRAY'):
    if color_space == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    elif color_space == 'RGB':
        return img

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
    vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

if __name__ == "__main__":
    # plot example from car and not-car dataset
    car_imgs = glob.glob("vehicles/**/*.png")
    notcar_imgs = glob.glob("non-vehicles/**/*.png")

    car_img = readImage(car_imgs[0])
    notcar_img = readImage(notcar_imgs[0])

    sns.set_style("white")
    fig = plt.figure(1, figsize=(6,3))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(car_img)
    ax1.set_title("Car")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax2 = fig.add_subplot(122)
    ax2.imshow(notcar_img)
    ax2.set_title("Not a car")
    plt.setp(ax2.get_yticklabels(), visible=0)
    ax2.set_xlabel("$x$")
    plt.tight_layout()
    plt.show()
    plt.savefig("output_images/figure01.png")

    # try hog on different color spaces and hog's parameters
    gray = cvtImage(car_img, 'GRAY')
    orientations_s = [3, 5, 7, 9]
    pixels_cell_s = [2, 4, 8, 16]
    cells_block = 2

    fig = plt.figure(2, figsize=(12,12))
    fig.clf()

    i = 1
    for orientations in orientations_s:
        for pixels_cell in pixels_cell_s:

            fv, hi = get_hog_features(gray, orientations,
                    pixels_cell, cells_block, vis=True, feature_vec=True)

            ax = fig.add_subplot(4,4,i)
            ax.imshow(hi, cmap='gray')
            plt.setp(ax.get_xticklabels(), visible=0)
            plt.setp(ax.get_yticklabels(), visible=0)

            if i <= 4:
                ax.set_title('%sppc'%pixels_cell)
            if i in [1, 5, 9, 13]:
                ax.set_ylabel('%s orientations'%orientations)

            i+=1

    plt.tight_layout()
    plt.show()
    plt.savefig("output_images/figure02.png")
