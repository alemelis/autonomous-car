import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FIND CORNERS
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# load example image
example_img = mpimg.imread("camera_cal/calibration1.jpg")
example_img_gray = cv2.cvtColor(example_img, cv2.COLOR_RGB2GRAY)
ret, corners = cv2.findChessboardCorners(example_img_gray, (9,5), None)

# find cornes in the example image
sns.set_style("white")
fig = plt.figure(1, figsize=(5,3))
fig.clf()
ax1 = fig.add_subplot(111)
cv2.drawChessboardCorners(example_img_gray, (9,5), corners, ret)
ax1.imshow(example_img)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
plt.tight_layout()
plt.show()
fig.savefig("output_images/fig0.png")

def preprocessImage(img_name):
    # read calibration image
    img = mpimg.imread(img_name)

    # convert to grayscale
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def findCalibrationCorners(img_name, cb_dim=(9,5)):
    gray = preprocessImage(img_name)

    # find corners
    return cv2.findChessboardCorners(gray, cb_dim, None)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CAMERA CALIBRATION
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# chessboard corners dimensions
chessboard_dims = (9,6)

img_pts = [] # image points list
obj_pts = [] # object points list

# initialise generic object point
obj_pt = np.zeros((chessboard_dims[0]*chessboard_dims[1], 3), np.float32)
obj_pt[:,:2] = np.mgrid[0:chessboard_dims[0],
                        0:chessboard_dims[1]].T.reshape(-1,2)

# process the calibration imagesn and extract corners
print("Processing calibration images")
for i in range(1, 21):
    print('#', sep='', end='', flush=1)
    img_name = "camera_cal/calibration{}.jpg".format(i)

    ret, corners = findCalibrationCorners(img_name, chessboard_dims)
    if ret:
        img_pts.append(corners)
        obj_pts.append(obj_pt)

print("Camera calibration")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
    (example_img.shape[0],example_img.shape[1]), None, None)

# compute the undistorted image
example_img = preprocessImage("camera_cal/calibration1.jpg")
undst_example_img = cv2.undistort(example_img, mtx, dist)

# show distorted and undistorted images
fig = plt.figure(2, figsize=(9,3))
fig.clf()
ax1 = fig.add_subplot(121)
ax1.set_title('Distorted')
ax2 = fig.add_subplot(122)
ax2.set_title('Undistorted')
plt.setp(ax2.get_yticklabels(), visible=0)
ax1.imshow(example_img, cmap='gray')
ax2.imshow(undst_example_img, cmap='gray')
ax2.set_xlabel("$x$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
plt.tight_layout()
plt.show()
fig.savefig("output_images/fig1.png")
