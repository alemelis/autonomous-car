import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from s2_perspective_transform import warpImage
from s4_finding_lines import findInitialLanes
from moviepy.editor import VideoFileClip

def computeCurvature(ys, left_fit, right_fit):
    y_eval = np.max(ys)
    left_curverad = ((1 + (2*left_fit[0]*y_eval +
        left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval +
        right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad

def loadCalibrationVariables():
    mtx  = np.loadtxt("data/mtx.dat")
    dist = np.loadtxt("data/dist.dat")
    M    = np.loadtxt("data/M.dat")
    offs = np.loadtxt("data/offset.dat")
    Minv = np.loadtxt("data/Minv.dat")

    return mtx, dist, M, offs, Minv

def colorThresholding(img_name, mtx, dist, M, thresholds=(60, 255)):
    # preprocess image
    warped = warpImage(img_name, mtx, dist, M, hls=1)

    # thresholding
    img_bin = np.zeros_like(warped)
    img_bin[(warped >= thresholds[0]) & (warped <= thresholds[1])] = 1

    return img_bin

def gradThresholding(img_name, mtx, dist, M, thresholds=(20, 100)):
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

def mixThresholding(img_name, mtx, dist, M,
        color_ths=(60, 255), sobel_ths=(20,100)):
    color_bin = colorThresholding(img_name, mtx, dist, M, color_ths)
    sobel_bin = gradThresholding(img_name, mtx, dist, M, sobel_ths)

    mix_bin = np.zeros_like(color_bin)
    mix_bin[(color_bin == 1) | (sobel_bin == 1)] = 1

    return mix_bin

def processFrame(img_name, windows=10, w_half_width=100, min_pix=50):
    global mtx, dist, M
    th_img = mixThresholding(img_name, mtx, dist, M)

    nnzr = th_img.nonzero()
    nnzr_y = np.array(nnzr[0])
    nnzr_x = np.array(nnzr[1])

    global p_L, p_R, ys
    lane_idxs_L = ((nnzr_x > (p_L[0]*(nnzr_y**2) +
        p_L[1]*nnzr_y + p_L[2] - w_half_width)) &
        (nnzr_x < (p_L[0]*(nnzr_y**2) + p_L[1]*nnzr_y + p_L[2] + w_half_width)))
    lane_idxs_R = ((nnzr_x > (p_R[0]*(nnzr_y**2) +
        p_R[1]*nnzr_y + p_R[2] - w_half_width)) &
        (nnzr_x < (p_R[0]*(nnzr_y**2) + p_R[1]*nnzr_y + p_R[2] + w_half_width)))

    # Again, extract left and right line pixel positions
    x_L = nnzr_x[lane_idxs_L]
    y_L = nnzr_y[lane_idxs_L]
    x_R = nnzr_x[lane_idxs_R]
    y_R = nnzr_y[lane_idxs_R]

    # Fit a second order polynomial to each
    p_L = np.polyfit(y_L, x_L, 2)
    p_R = np.polyfit(y_R, x_R, 2)

    x_fit_L = p_L[0]*ys**2 + p_L[1]*ys + p_L[2]
    x_fit_R = p_R[0]*ys**2 + p_R[1]*ys + p_R[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([x_fit_L-w_half_width, ys]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([x_fit_L+w_half_width, ys])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([x_fit_R-w_half_width, ys]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([x_fit_R+w_half_width, ys])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(th_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([x_fit_L, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([x_fit_R, ys])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    global Minv
    newwarp = cv2.warpPerspective(color_warp, Minv, (th_img.shape[1],
        th_img.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(img_name, 1, newwarp, 0.3, 0)

mtx, dist, M, offs, Minv = loadCalibrationVariables()
project_clip = VideoFileClip("videos/challenge_video.mp4")
for frame in project_clip.iter_frames():
    (current_L, current_R, lane_idxs_L,
        lane_idxs_R, p_L, p_R) = findInitialLanes(frame)
    break
ys = np.linspace(0, frame.shape[0]-1, frame.shape[0])

prj_clip = project_clip.fl_image(processFrame)
prj_clip.write_videofile("project.mp4", audio=False)
