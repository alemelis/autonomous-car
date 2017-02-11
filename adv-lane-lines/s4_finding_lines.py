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

    return current_L, current_R, lane_idxs_L, lane_idxs_R, p_L, p_R

if __name__ == "__main__":
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # HISTOGRAM
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    img_name = "test_images/test2.jpg"
    th_img = mixThresholding(img_name)

    histogram = np.sum(th_img[np.int(th_img.shape[0]/2):,:], axis=0)

    # plot histogram
    sns.set_style("white")
    fig = plt.figure(1, figsize=(12,3))
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(histogram)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Count")
    ax1.set_aspect('equal')
    plt.xlim(0,len(histogram))
    plt.ylim(0, np.max(histogram)+10)
    fig.set_tight_layout(1)
    plt.show()
    plt.savefig("output_images/fig7.png")

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # SLIDING WINDOWS
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

    fig = plt.figure(2, figsize=(6,4))
    fig.clf()
    ax1 = fig.add_subplot(111)
    plt.imshow(out_img)
    plt.plot(x_fit_L, ys, color='yellow')
    plt.plot(x_fit_R, ys, color='yellow')
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$x$")
    ax1.set_xlim(0, out_img.shape[1])
    ax1.set_ylim(out_img.shape[0],0)
    fig.set_tight_layout(1)
    plt.show()
    plt.savefig("output_images/fig8.png")

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # CURVATURES AND OFFSET
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # conversion ratios pixel-meters
    y_m_p = 30./281 # meters per pixel y-wise
    x_m_p = 3.7/720 # x-wise

    # fit to meters
    p_L_m = np.polyfit(y_L * y_m_p, x_L * x_m_p, 2)
    p_R_m = np.polyfit(y_R* y_m_p, x_R* x_m_p, 2)

    # curvature radii
    ys_m = ys*y_m_p
    R_L = ( 1 + ( 2*p_L_m[0]*ys_m + p_L_m[1] )**1.5) / np.absolute(2*p_L_m[0])
    R_R = ( 1 + ( 2*p_R_m[0]*ys_m + p_R_m[1] )**1.5) / np.absolute(2*p_R_m[0])

    # center offset
    x_R_b = p_R_m[0]*ys_m[-1]**2 + p_R_m[1]*ys_m[-1] + p_R_m[2]
    x_L_b = p_L_m[0]*ys_m[-1]**2 + p_L_m[1]*ys_m[-1] + p_L_m[2]
    center_offset = ((x_R_b-x_L_b)*0.5 + x_L_b) - out_img.shape[1]*0.5*x_m_p

    print("left curvature radius: %5.2fm"%R_L[-1])
    print("right curvature radius: %5.2fm"%R_L[-1])
    print("center offset: %5.2fm"%center_offset)

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # PLOT LANE ON ORIGINAL IMAGE (from lecture notes)
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    image = mpimg.imread(img_name)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(th_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([x_fit_L, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([x_fit_R, ys])))])
    pts = np.hstack((pts_left, pts_right))

    # draw the polygon
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse
    # perspective matrix (Minv)
    M = np.loadtxt("data/M.dat")
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv,
        (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    fig = plt.figure(3, figsize=(6,4))
    fig.clf()
    ax1 = fig.add_subplot(111)
    plt.imshow(result)
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$x$")
    fig.set_tight_layout(1)
    plt.show()
    plt.savefig("output_images/fig9.png")
