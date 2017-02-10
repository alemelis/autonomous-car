import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from s3_thresholding import mixThresholding

def findInitialLanes(img_name, windows=10, w_half_width=100, min_pix=50):
    # load and process image
    th_img = mixThresholding(img_name)

    # compute histogram on the lower half
    histogram = np.sum(th_img[np.int(th_img.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((th_img, th_img, th_img))*255

    # split the histogram and find the right and left starting pixels
    midpoint = np.int(histogram.shape[0]/2)
    base_L = np.argmax(histogram[:midpoint])
    base_R = np.argmax(histogram[midpoint:])+midpoint

    # set current position
    current_L = base_L
    current_R = base_R

    # define windows height
    w_height = np.int(th_img.shape[0]/windows)

    # nonzero pixels coordinates
    nnzr = th_img.nonzero()
    nnzr_y = np.array(nnzr[0])
    nnzr_x = np.array(nnzr[1])

    # lane coordinates lists
    lane_idxs_L = []
    lane_idxs_R = []

    # loop over the windows
    for i in range(windows):

        # window coordinates
        P34_y = th_img.shape[0] - (i+1)*w_height
        P12_y = th_img.shape[0] - i*w_height

        P14_x_L = current_L - w_half_width
        P23_x_L = current_L + w_half_width

        P14_x_R = current_R - w_half_width
        P23_x_R = current_R + w_half_width

        # find non-zero pixels in the two windows
        nnzr_L = ((nnzr_y >= P34_y)   & (nnzr_y < P12_y) &
                  (nnzr_x >= P14_x_L) & (nnzr_x < P23_x_L)).nonzero()[0]
        nnzr_R = ((nnzr_y >= P34_y)   & (nnzr_y < P12_y) &
                  (nnzr_x >= P14_x_R) & (nnzr_x < P23_x_R)).nonzero()[0]
        lane_idxs_L.append(nnzr_L)
        lane_idxs_R.append(nnzr_R)

        # find new position if enough pixels are found
        if len(nnzr_L) > min_pix:
            current_L = np.int(np.mean(nnzr_x[nnzr_L]))

        if len(nnzr_R) > min_pix:
            current_R = np.int(np.mean(nnzr_x[nnzr_R]))
    lane_idxs_L = np.concatenate(lane_idxs_L)
    lane_idxs_R = np.concatenate(lane_idxs_R)

    # extract lines position
    x_L = nnzr_x[lane_idxs_L]
    y_L = nnzr_y[lane_idxs_L]

    x_R = nnzr_x[lane_idxs_R]
    y_R = nnzr_y[lane_idxs_R]

    # fit polynomials
    p_L = np.polyfit(y_L, x_L, 2)
    p_R = np.polyfit(y_R, x_R, 2)

    ys = np.linspace(0, th_img.shape[0]-1, th_img.shape[0] )
    x_fit_L = p_L[0]*ys**2 + p_L[1]*ys + p_L[2]
    x_fit_R = p_R[0]*ys**2 + p_R[1]*ys + p_R[2]

    # out_img[nnzr_y[lane_idxs_L], nnzr_x[lane_idxs_L]] = [255, 0, 0]
    # out_img[nnzr_y[lane_idxs_R], nnzr_x[lane_idxs_R]] = [0, 0, 255]

    return current_L, current_R, lane_idxs_L, lane_idxs_R, p_L, p_R

if __name__ == "__main__":
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # FINDING LINES
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    img_name = "test_images/test2.jpg"
    th_img = mixThresholding(img_name)

    histogram = np.sum(th_img[np.int(th_img.shape[0]/2):,:], axis=0)

    # # plot histogram
    # sns.set_style("white")
    # fig = plt.figure(1, figsize=(12,3))
    # fig.clf()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(histogram)
    # ax1.set_xlabel("$x$")
    # ax1.set_ylabel("Count")
    # ax1.set_aspect('equal')
    # plt.xlim(0,len(histogram))
    # plt.ylim(0, np.max(histogram)+10)
    # fig.set_tight_layout(1)
    # plt.show()
    # plt.savefig("output_images/fig7.png")

    #$$$


    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((th_img, th_img, th_img))*255

    # split the histogram and find the right and left starting pixels
    midpoint = np.int(histogram.shape[0]/2)
    base_L = np.argmax(histogram[:midpoint])
    base_R = np.argmax(histogram[midpoint:])+midpoint

    # set current position
    current_L = base_L
    current_R = base_R

    # define windows number and shape
    windows = 10
    w_height = np.int(th_img.shape[0]/windows)
    w_half_width = 100

    # nonzero pixels coordinates
    nnzr = th_img.nonzero()
    nnzr_y = np.array(nnzr[0])
    nnzr_x = np.array(nnzr[1])

    # minimum number of pixels found to recenter window
    min_pix = 50

    # lane coordinates lists
    lane_idxs_L = []
    lane_idxs_R = []

    # loop over the windows
    for i in range(windows):

        # window coordinates
        P34_y = th_img.shape[0] - (i+1)*w_height
        P12_y = th_img.shape[0] - i*w_height

        P14_x_L = current_L - w_half_width
        P23_x_L = current_L + w_half_width

        P14_x_R = current_R - w_half_width
        P23_x_R = current_R + w_half_width

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (P14_x_L, P34_y), (P23_x_L, P12_y), (0,255,0), 2)
        cv2.rectangle(out_img, (P14_x_R, P34_y), (P23_x_R, P12_y), (0,255,0), 2)

        # find non-zero pixels in the two windows
        nnzr_L = ((nnzr_y >= P34_y)   & (nnzr_y < P12_y) &
                  (nnzr_x >= P14_x_L) & (nnzr_x < P23_x_L)).nonzero()[0]
        nnzr_R = ((nnzr_y >= P34_y)   & (nnzr_y < P12_y) &
                  (nnzr_x >= P14_x_R) & (nnzr_x < P23_x_R)).nonzero()[0]
        lane_idxs_L.append(nnzr_L)
        lane_idxs_R.append(nnzr_R)

        # find new position if enough pixels are found
        if len(nnzr_L) > min_pix:
            current_L = np.int(np.mean(nnzr_x[nnzr_L]))

        if len(nnzr_R) > min_pix:
            current_R = np.int(np.mean(nnzr_x[nnzr_R]))
    lane_idxs_L = np.concatenate(lane_idxs_L)
    lane_idxs_R = np.concatenate(lane_idxs_R)

    # extract lines position
    x_L = nnzr_x[lane_idxs_L]
    y_L = nnzr_y[lane_idxs_L]

    x_R = nnzr_x[lane_idxs_R]
    y_R = nnzr_y[lane_idxs_R]

    # fit polynomials
    p_L = np.polyfit(y_L, x_L, 2)
    p_R = np.polyfit(y_R, x_R, 2)

    ys = np.linspace(0, th_img.shape[0]-1, th_img.shape[0] )
    x_fit_L = p_L[0]*ys**2 + p_L[1]*ys + p_L[2]
    x_fit_R = p_R[0]*ys**2 + p_R[1]*ys + p_R[2]

    out_img[nnzr_y[lane_idxs_L], nnzr_x[lane_idxs_L]] = [255, 0, 0]
    out_img[nnzr_y[lane_idxs_R], nnzr_x[lane_idxs_R]] = [0, 0, 255]

    # plot
    plt.imshow(out_img)
    plt.plot(x_fit_L, ys, color='yellow')
    plt.plot(x_fit_R, ys, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
