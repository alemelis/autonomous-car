import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from s2_perspective_transform import warpImage
from s4_finding_lines import findInitialLanes

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

def findLines(nnzr, w_half_width):
    nnzr_y = np.array(nnzr[0])
    nnzr_x = np.array(nnzr[1])

    global p_L, p_R, ys
    lane_idxs_L = ((nnzr_x > (p_L[0]*(nnzr_y**2) +
                              p_L[1]*nnzr_y +
                              p_L[2] - w_half_width)) &
                   (nnzr_x < (p_L[0]*(nnzr_y**2) +
                              p_L[1]*nnzr_y +
                              p_L[2] + w_half_width)))

    lane_idxs_R = ((nnzr_x > (p_R[0]*(nnzr_y**2) +
                              p_R[1]*nnzr_y +
                              p_R[2] - w_half_width)) &
                   (nnzr_x < (p_R[0]*(nnzr_y**2) +
                              p_R[1]*nnzr_y +
                              p_R[2] + w_half_width)))
    x_L = nnzr_x[lane_idxs_L]
    y_L = nnzr_y[lane_idxs_L]
    x_R = nnzr_x[lane_idxs_R]
    y_R = nnzr_y[lane_idxs_R]

    # Fit a second order polynomial to each
    p_L_ = np.polyfit(y_L, x_L, 2)
    p_R_ = np.polyfit(y_R, x_R, 2)

    x_fit_L_ = p_L_[0]*ys**2 + p_L_[1]*ys + p_L_[2]
    x_fit_R_ = p_R_[0]*ys**2 + p_R_[1]*ys + p_R_[2]

    R_L, R_R, center_offset = computeCurvature(y_L, x_L, y_R, x_R)

    return p_L_, p_R_, x_fit_L_, x_fit_R_, R_L, R_R, center_offset

# this function code was mostly taken from the lecture notes
def drawPolygonOnRoad(img, th_img, x_fit_L, x_fit_R, w_half_width,
                        R_L, R_R, center_offset):
    global ys
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

    # create a new image to draw on
    warp_zero = np.zeros_like(th_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([x_fit_L, ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([x_fit_R, ys])))])
    pts = np.hstack((pts_left, pts_right))

    # draw the polygon
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # write information on screen
    # curvature radii
    cv2.putText(img, "left radius: %5.2fm"%R_L[-1], (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    cv2.putText(img, "right radius: %5.2fm"%R_R[-1], (10,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

    global x_m_p
    cv2.putText(img, "offset: %5.2fm"%((center_offset-img.shape[1]*0.5)*x_m_p),
                (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

    # Unwarp the image usign the inverse camera matrix
    global Minv
    newwarp = cv2.warpPerspective(color_warp, Minv, (th_img.shape[1],
        th_img.shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def computeCurvature(left_y, left_x, right_y, right_x):
    global y_m_p, x_m_p, frame_width
    p_L = np.polyfit(left_y * y_m_p, left_x * x_m_p, 2)
    p_R = np.polyfit(right_y* y_m_p, right_x* x_m_p, 2)

    x_R_b = p_R[0]*ys_m[-1]**2 + p_R[1]*ys_m[-1] + p_R[2]
    x_L_b = p_L[0]*ys_m[-1]**2 + p_L[1]*ys_m[-1] + p_L[2]
    center_offset = ((x_R_b-x_L_b)*0.5 + x_L_b)/x_m_p

    R_L = ( 1 + ( 2*p_L[0]*ys_m + p_L[1] )**1.5) / np.absolute(2*p_L[0])
    R_R = ( 1 + ( 2*p_R[0]*ys_m + p_R[1] )**1.5) / np.absolute(2*p_R[0])

    return R_L, R_R, center_offset

def processFrame(img, windows=10, w_half_width=100, min_pix=50):
    global mtx, dist, M
    th_img = mixThresholding(img, mtx, dist, M)

    # find lines in the current frame
    nnzr = th_img.nonzero()
    global p_L, p_R, x_fit_L, x_fit_R, R_R, R_L
    (p_L_, p_R_, x_fit_L_, x_fit_R_,
        R_L_, R_R_, center_offset) = findLines(nnzr, w_half_width)

    # check the line
    global p_Ls, p_Rs, R_Ls, R_Rs, R_R, R_L

    # for the first 15 frames just take the lines for good and append
    if len(p_Ls) <= 15:
        p_Ls.append(p_L_)
        p_Rs.append(p_R_)
        if str(R_L_[-1]) != "nan":
            R_L = R_L_
            R_Ls.append(R_L_)
        if str(R_R_[-1]) != "nan":
            R_R = R_R_
            R_Rs.append(R_R_)
    else:
        # in the successive frames, check if the lines ave roughly the same
        # curvature. If the curvature differs for more than the 50%, discard
        # both lines and use the previously computed
        if (all(np.absolute(R_L*100/R_R-100)) < 50):
            p_L = p_L_
            p_R = p_R_
            x_fit_L = x_fit_L_
            x_fit_R = x_fit_R_
            if str(R_L_[-1]) != "nan":
                R_L = R_L_
                R_Ls.append(R_L_)
            if str(R_R_[-1]) != "nan":
                R_R = R_R_
                R_Rs.append(R_R_)

        # take rid of the oldes line (first in the list)
        p_Ls = p_Ls[1:]
        p_Rs = p_Rs[1:]
        R_Ls = R_Ls[1:]
        R_Rs = R_Rs[1:]

        # append the current fit to the listss
        p_Ls.append(p_L)
        p_Rs.append(p_R)
        R_Ls.append(R_L)
        R_Rs.append(R_R)

    # take current values as the mean from the previous 15 frames (about 1s)
    p_L = np.mean(np.array(p_Ls), 0)
    x_fit_L = p_L_[0]*ys**2 + p_L_[1]*ys + p_L_[2]

    p_R = np.mean(np.array(p_Rs), 0)
    x_fit_R_ = p_R_[0]*ys**2 + p_R_[1]*ys + p_R_[2]

    R_L_m = np.mean(np.array(R_Ls), 0)
    R_R_m = np.mean(np.array(R_Rs), 0)

    # draw polygon on the video frame
    return drawPolygonOnRoad(img, th_img, x_fit_L, x_fit_R,
                                w_half_width, R_L_m, R_R_m, center_offset)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    # constants definition
    mtx, dist, M, offs, Minv = loadCalibrationVariables()

    y_m_p = 30./281 # meters per pixel y-wise
    x_m_p = 3.7/720 # x-wise

    # process first frame with sliding window methods
    project_clip = VideoFileClip("videos/project_video.mp4")
    for frame in project_clip.iter_frames():
        (current_L, current_R, lane_idxs_L,
            lane_idxs_R, p_L, p_R) = findInitialLanes(frame)
        break

    # initial state
    # these are used as global variables in the functions above
    frame_width = frame.shape[1]
    ys = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    ys_m = ys*y_m_p # convert to meter
    x_fit_L = p_L[0]*ys**2 + p_L[1]*ys + p_L[2]
    x_fit_R = p_R[0]*ys**2 + p_R[1]*ys + p_R[2]
    p_Ls = [p_L]
    p_Rs = [p_R]
    R_Ls = []
    R_Rs = []

    prj_clip = project_clip.fl_image(processFrame)
    prj_clip.write_videofile("project_out.mp4", audio=False)
