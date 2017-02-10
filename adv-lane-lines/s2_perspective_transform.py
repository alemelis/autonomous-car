import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import glob
from s1_calibration import undistortImage

def warpImage(image_name, mtx, dist, M, hls=0):
    und_img = undistortImage(image_name, mtx, dist, hls)

    img_size = (und_img.shape[1], und_img.shape[0])
    return cv2.warpPerspective(und_img, M, img_size, flags=cv2.INTER_LINEAR)

if __name__ == "__main__":
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # PERSPECTIVE TRANSFORM
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    print("Load calibration data")
    mtx = np.loadtxt("data/mtx.dat")
    dist = np.loadtxt("data/dist.dat")

    print("Undistort source image")
    src_name = "test_images/straight_lines1.jpg"
    undst_src = undistortImage(src_name, mtx, dist)

    # source points picked by hand on the source image
    P1 = [569,  468]
    P2 = [716,  468]
    P3 = [1122, 720]
    P4 = [198,  720]
    src_pts = np.float32([P1, P2, P3, P4])

    # destination points defined by means of offset value
    img_size = (undst_src.shape[1],undst_src.shape[0])
    offset = 350
    np.savetxt("data/offset.dat", offset) # save for next steps
    dst_pts = np.float32([[offset,             offset],
                          [img_size[0]-offset, offset],
                          [img_size[0]-offset, img_size[1]],
                          [offset,             img_size[1]]])

    print("Transform perspective")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    np.savetxt("data/M.dat", M)

    print("Warp image")
    warped = cv2.warpPerspective(undst_src, M, img_size, flags=cv2.INTER_LINEAR)

    # plot images
    sns.set_style("white")
    fig = plt.figure(1, figsize=(9,3))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(undst_src, cmap='gray')
    ax1.scatter(src_pts[:,0], src_pts[:,1], marker='o',
        lw=3, s=70, edgecolor='crimson', color="w", zorder=10)
    src_ = np.vstack([src_pts, src_pts[0,:]]) # dummy array for plotting purposes
    ax1.plot(src_[:,0], src_[:,1], '--', lw=3, color='crimson')
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_xlim(0, undst_src.shape[1])
    ax1.set_ylim(undst_src.shape[0],0)
    ax1.set_title("Undistorted")
    ax2 = fig.add_subplot(122)
    ax2.imshow(warped, cmap='gray')
    iw = undst_src.shape[1]
    ih = undst_src.shape[0]
    ax2.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.set_xlabel("$x$")
    ax2.set_title("Warped")
    plt.setp(ax2.get_yticklabels(), visible=0)
    ax2.set_xlim(0, warped.shape[1])
    ax2.set_ylim(warped.shape[0],0)
    plt.tight_layout()
    plt.show()
    plt.savefig("output_images/fig2.png")

    # apply pipeline to curved lines image
    crv_name = "test_images/test2.jpg"
    crv_img = mpimg.imread(crv_name)
    crv_warped = warpImage(crv_name, mtx, dist, M)

    fig = plt.figure(2, figsize=(9,3))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(crv_img)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_xlim(0, undst_src.shape[1])
    ax1.set_ylim(undst_src.shape[0],0)
    ax1.set_title("Original")
    ax2 = fig.add_subplot(122)
    ax2.imshow(crv_warped, cmap='gray')
    ax2.vlines(offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.vlines(iw-offset, 0, iw+50, linestyles='--', color="crimson", lw=3)
    ax2.set_xlabel("$x$")
    ax2.set_title("Warped")
    plt.setp(ax2.get_yticklabels(), visible=0)
    ax2.set_xlim(0, warped.shape[1])
    ax2.set_ylim(warped.shape[0],0)
    plt.tight_layout()
    plt.show()
    plt.savefig("output_images/fig3.png")
