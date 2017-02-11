# Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/fig0.png "Corners"
[image1]: ./output_images/fig1.png "Distorted and Undistorted"
[image2-0]: ./output_images/fig2-0.png "Select source points"
[image2]: ./output_images/fig2.png "Straight lines warped"
[image3]: ./output_images/fig3.png "Curved lines warped"
[image4]: ./output_images/fig4.png "Saturation channel thresholding"
[image5]: ./output_images/fig5.png "x-wise gradient thresholding"
[image6]: ./output_images/fig6.png "Combined thresholding"
[image7]: ./output_images/fig7.png "Histogram"
[image8]: ./output_images/fig8.png "Sliding windows method"
[image9]: ./output_images/fig9.png "Lane found"
[image10]: ./output_images/fig10.png "Challenge failed"

#### Camera Calibration

The code for this step is contained in `s1_calibration.py` file. The computed calibration matrix `mtx.dat` and distortion coefficients `dist.dat` are saved in `./data/` folder.

The camera calibration was done by processing chessboard images taken from different angles. The images were imported from `camera_cal/` by means of `glob` library (L 72).
Then, the calibration pipeline consisted of the following steps.

* Prepare a list of _object points_, i.e., the 3D coordinates of the real undistorted chessboard corners. These are the same for each calibration images. Therefore, an array `obj_pt` containing the coordinates for one image was generated once and then used to populate the list `obj_pts`. (L 63:69)

* The coordinates of the chessboard corners in the 2D image, the _image points_, were automatically extracted from each image by means of `findCalibrationCorners()` function (L 75). These corners were appended to `img_pts` list. The function takes grayscale images as input, hence all the calibration images were first converted (`preprocessImage()`).

* `img_pts` and `obj_pts` were used to compute the camera calibration matrix and the distortion coefficients through `cv2.calibrateCamera()`. The images can also be visualised in the undistorted form by applying `undistortImage()`. (L 67, 75)

_Distorted with corners extracted, and undistorted form of the first calibration image_

![alt text][image1]

#### Perspective transform

The code for my perspective transform is in `s2_perspective_transform.py`. It includes the function `warpImage()` (L 9). This function takes as input the image name as a string and the `mtx`, `dist`, and `M` parameters. The first two, can be loaded from `./data/` as they were computed in the previous step. `M` is the perspective transform matrix. This was computed in the following steps:

* Compute the undistorted form of the image with functions from `s1_calibration.py` (L19:25).

* Select by hand the source points `P1`, `P2`, `P3`, `P4` and store them in the array `src_pts`. This was done on pyplot interactive window (L 27:32)
![alt text][image2-0]

* Define the destination points `dst_pts` in function of the image size (L 34:41).

* Get the perspective transform matrix `M` by means of `cv2.getPerspectiveTransform()` and save `M` in `./data/M.dat` (L 43:45).

* Warp image to obtain the bird's-eye view (L 48)

  _(left) Undistorted test image with straight lines. The source points are marked. (right) Warped image; the destination points are indicated by the dashed line. The lane lines are parallel to the dashed lines._

  ![alt text][image2]

* Apply the pipeline to an image with curved lines

  _Original curved lines image and warped version. The lane lines are not parallel to the dashed lines in the bird's-eye view._

  ![alt text][image3]

#### Thresholding

Two thresholding methods were employed (see `s3_thresholding.py`):

* __Color thresholding__
  * The saturation channel was extracted from the RGB image. This was observed to be the most robust channel upon which perform the color thresholding (L 65).
  * The S channel image was warped through `warpImage()` and thresholded by means of two values stored in the tuple `S_thresholds`. These values were selected by trial/error on the test image (L 72).

  ![alt text][image4]


* __Gradient thresholding__
  * Convert the image to grayscale and warp to fix the perspective (L 107).
  * Compute x-wise gradient by means of `cv2.Sobel()` (L 110:111).
  * Take the absolute values and scale the gradient image (L 114).
  * Apply Sobel thresholds (L 117:120).

  ![alt text][image5]

The combined thresholding was obtained by operating the _union_ of the two binary images (L 154):
```python
mix_bin[(color_th == 1) | (sobel_th == 1)] = 1
```

_Combined thresholding pipeline. The resulting image is on the bottom right._

![alt text][image6]

#### Lines detection
Thi section code is in `s4_finding_lines.py`. Lines were detected by using the _sliding window_ method (L 83:189):

* Split the image horizontally and take the lower half. This is the image part where the lines are wide apart and close to the camera (L 89).

* Plot a pixel intensity histogram and split vertically. Take the two peaks in the two sides, as these indicate where the lines are most likely to be in the image (L 91:116).
![alt text][image7]

* Define the number and the size of the sliding windows (code mostly from lecture notes, L 122:125). Then, loop through all of them to:
  * identify the window boundaries (L 142:154);
  * select all the non-zero pixels inside the window and save their indices in a list (L 156:162).

  The two lists of pixels can be used to fit a second order polynomial function with `np.polyfit()` (L 173:189).
![alt text][image8]

The lines curvature (L 209:219) was calculated as

![img](http://i.imgur.com/nwcqim9.png)

and expressed in meters by using the two conversion ratios
```python
y_m_p = 30./281 # meters per pixel y-wise
x_m_p = 3.7/720 # x-wise
```
The offset of the car from the lane center (L 222:224) was computed as
```python
center_offset = ((x_R_b-x_L_b)*0.5 + x_L_b) - out_img.shape[1]*0.5*x_m_p
```
where `x_R_b` and `x_L_b` are the lines position at the bottom of the image.

In the example above, it resulted
```
left curvature radius: 2418.65m
right curvature radius: 2418.65m
center offset:  0.22m
```

The lane can be plotted back on the original image by means of the code provided in the notes. In particular, the lane is first drawn on the warped image and then un-warped by applying the camera matrix inverse `Minv = np.linalg.inv(M)`. The resulting image is reported below.

![alt text][image9]

### Pipeline (video)

Here's a [link to my video result](https://www.youtube.com/watch?v=NcVSKMoLsZY), which is attached also as `project_out.mp4`

The video was processed with the functions in `s5_find_lane_on_video.py`. There, the main functions defined in the previous steps were reported.

First, the sliding window method (`finInitialLanes()`) is used to identify the lines in the first video frame (L 219:243). The polynomial fits are then passed to `processFrame()` as global variables. In `processFrame()`, the lines are found by using the previous position as starting points for the pixel search (`findLines()`). If the newly found lines have similar curvature radius (these must differ for less than 50%), they are taken as good (L 163:199). In order to smooth the lines, the results to be plotted are averaged with the previous _15_ valid lines (L 201:209). The lane, curvature radii, and the center offset are plotted back on the frame (`drawPolygonOnRoad()`).

---

### Discussion

The method as it is seems to work well on the test video,

![img](http://i.giphy.com/26xBBfiPM9aJ3D6LK.gif)

even on the parts where a sudden change in the pavement occurs

![img](http://i.giphy.com/26gs7DEEfNRL9pVLi.gif)

The lane is pretty stable and does not wobble too much because of the dashed line. An increase in stability could be achieved by applying a Gaussian blur before the thresholding step.

The pipeline fails on the challenge video right at the beginning, when the sliding window method takes part.

![alt text][image10]

The strong shadow on the left of the lane is mistaken for a line. This may be solved by masking the image before the thresholding step. Also, the curvature is over-estimated and the lines intersect at one end.

The pipeline can be made more robust by adding more controls to the lines. For example, a check on the distance between lines along the entire image (we assume that the lines are parallel). A more rigorous check may indicate that the pipeline needs to be restarted with a new windows search.
