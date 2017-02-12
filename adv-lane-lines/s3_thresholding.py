import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from s2_perspective_transform import warpImage

# load calibration variables
def loadCalibrationVariables():
    mtx  = np.loadtxt("data/mtx.dat")
    dist = np.loadtxt("data/dist.dat")
    M    = np.loadtxt("data/M.dat")
    offs = np.loadtxt("data/offset.dat")

    return mtx, dist, M, offs

def colorThresholding(img_name, thresholds=(60, 255)):
    mtx, dist, M, offset = loadCalibrationVariables()

    # preprocess image
    warped = warpImage(img_name, mtx, dist, M, hls=1)

    # thresholding
    img_bin = np.zeros_like(warped)
    img_bin[(warped >= thresholds[0]) & (warped <= thresholds[1])] = 1

    return img_bin

def gradThresholding(img_name, thresholds=(20, 100)):
    mtx, dist, M, offset = loadCalibrationVariables()

    # preprocess image
    warped = warpImage(img_name, mtx, dist, M)

    # compute x-wise gradient and take the absolute value
    sobel_x = cv2.Sobel(warped, cv2.CV_64F, 1, 0)
    sobel_x_abs = np.absolute(sobel_x)

    # scale the absolute values
    scaled = np.uint8(255*sobel_x_abs/np.max(sobel_x_abs))

    # apply thresholding
    sobel_bin = np.zeros_like(scaled)
    sobel_bin[(scaled >= thresholds[0]) & (scaled <= thresholds[1])] = 1

    return sobel_bin

def mixThresholding(img_name, color_ths=(60, 255), sobel_ths=(20,100)):
    color_bin = colorThresholding(img_name, color_ths)
    sobel_bin = gradThresholding(img_name, sobel_ths)

    mix_bin = np.zeros_like(color_bin)
    mix_bin[(color_bin == 1) | (sobel_bin == 1)] = 1

    return mix_bin

def trickyHSV(img_name):
    mtx, dist, M, offset = loadCalibrationVariables()
    # after the first review, the referee suggested this additional steps
    if type(img_name) == str:
        # read calibration image
        img = mpimg.imread(img_name)
    else:
        img = img_name

    und_img = cv2.undistort(img, mtx, dist)

    img_size = (und_img.shape[1], und_img.shape[0])
    warped = cv2.warpPerspective(und_img, M, img_size, flags=cv2.INTER_LINEAR)

    HSV = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)

    img_bin = np.zeros_like(HSV)[:,:,0]

    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(warped, (200,200,200), (255,255,255))

    bit_layer = img_bin | yellow | white | white_2 | white_3

    return bit_layer-1

if __name__ == "__main__":
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # COLOR THRESHOLDING
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    mtx, dist, M, offset = loadCalibrationVariables()

    # load, extract S channel, and warp example image
    img_name = "test_images/test2.jpg"
    img_warp = warpImage(img_name, mtx, dist, M, hls=1)

    # initialise binary image
    img_bin = np.zeros_like(img_warp)

    # select thresolds
    S_thresholds = (60, 255)
    img_bin[(img_warp >= S_thresholds[0]) & (img_warp <= S_thresholds[1])] = 1

    # plot images
    sns.set_style("white")
    fig = plt.figure(1, figsize=(9,3))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_warp, cmap='gray')
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_xlim(0, img_warp.shape[1])
    ax1.set_ylim(img_warp.shape[0],0)
    ax1.set_title("Warped S channel")

    ax2 = fig.add_subplot(122)
    ax2.imshow(img_bin, cmap='gray')
    iw = img_warp.shape[1]
    ih = img_warp.shape[0]
    ax2.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.set_xlabel("$x$")
    ax2.set_title("S channel thesholding")
    plt.setp(ax2.get_yticklabels(), visible=0)
    ax2.set_xlim(0, img_bin.shape[1])
    ax2.set_ylim(img_bin.shape[0],0)

    plt.tight_layout()
    plt.show()
    # plt.savefig("output_images/fig4.png")

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # GRADIENT THRESHOLDING
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # load, convert to grayscale, and warp example image
    gray_warp = warpImage(img_name, mtx, dist, M)

    # compute x-wise gradient and take the absolute value
    gray_Sx = cv2.Sobel(gray_warp, cv2.CV_64F, 1, 0)
    gray_Sx_abs = np.absolute(gray_Sx)

    # scale the absolute values
    scaled_Sx_abs = np.uint8(255*gray_Sx_abs/np.max(gray_Sx_abs))

    # apply thresholding
    Sx_th_min = 20
    Sx_th_max = 100
    Sx_bin = np.zeros_like(scaled_Sx_abs)
    Sx_bin[(scaled_Sx_abs >= Sx_th_min) & (scaled_Sx_abs <= Sx_th_max)] = 1

    # plot images
    sns.set_style("white")
    fig = plt.figure(2, figsize=(9,3))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(gray_warp, cmap='gray')
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_xlim(0, gray_warp.shape[1])
    ax1.set_ylim(gray_warp.shape[0],0)
    ax1.set_title("Warped grayscale images")

    ax2 = fig.add_subplot(122)
    ax2.imshow(Sx_bin, cmap='gray')
    iw = gray_warp.shape[1]
    ih = gray_warp.shape[0]
    ax2.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.set_xlabel("$x$")
    ax2.set_title("Sobel x thesholding")
    plt.setp(ax2.get_yticklabels(), visible=0)
    ax2.set_xlim(0, Sx_bin.shape[1])
    ax2.set_ylim(Sx_bin.shape[0],0)

    plt.tight_layout()
    plt.show()
    # plt.savefig("output_images/fig5.png")

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # COMBINE THRESHOLDING
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    mix_bin = np.zeros_like(gray_warp)
    mix_bin[(img_bin == 1) | (Sx_bin == 1)] = 1

    # plot images
    sns.set_style("white")
    fig = plt.figure(3, figsize=(9,6))
    fig.clf()
    ax1 = fig.add_subplot(221)
    ax1.imshow(gray_warp, cmap='gray')
    ax1.set_ylabel("$y$")
    plt.setp(ax1.get_xticklabels(), visible=0)
    ax1.set_xlim(0, gray_warp.shape[1])
    ax1.set_ylim(gray_warp.shape[0],0)
    ax1.set_title("Warped grayscale images")

    ax2 = fig.add_subplot(222)
    ax2.imshow(Sx_bin, cmap='gray')
    iw = gray_warp.shape[1]
    ih = gray_warp.shape[0]
    ax2.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.set_title("Sobel x thesholding")
    plt.setp(ax2.get_yticklabels(), visible=0)
    plt.setp(ax2.get_xticklabels(), visible=0)
    ax2.set_xlim(0, Sx_bin.shape[1])
    ax2.set_ylim(Sx_bin.shape[0],0)

    ax3 = fig.add_subplot(223)
    ax3.imshow(img_bin, cmap='gray')
    iw = img_warp.shape[1]
    ih = img_warp.shape[0]
    ax3.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax3.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$y$")
    ax3.set_title("S channel thesholding")
    ax3.set_xlim(0, img_bin.shape[1])
    ax3.set_ylim(img_bin.shape[0],0)

    ax4 = fig.add_subplot(224)
    ax4.imshow(mix_bin, cmap='gray')
    iw = gray_warp.shape[1]
    ih = gray_warp.shape[0]
    ax4.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax4.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax4.set_title("Combined thesholding")
    ax4.set_xlabel("$x$")
    plt.setp(ax4.get_yticklabels(), visible=0)
    ax4.set_xlim(0, mix_bin.shape[1])
    ax4.set_ylim(mix_bin.shape[0],0)

    plt.tight_layout()
    plt.show()
    # plt.savefig("output_images/fig6.png")

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # COMBINE THRESHOLDING IN OTHER COLOR SPACES (suggested by a reviewer)
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    b = trickyHSV(img_name)

    fig = plt.figure(4, figsize=(5,3))
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.imshow(b)
    ax1.set_title("Combined thesholding HSV")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    plt.tight_layout()
    plt.show()
    plt.savefig("output_images/fig-R1.png")
