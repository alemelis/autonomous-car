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

<!-- [image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video" -->

[image0]: ./output_images/fig0.png "Corners"
[image1]: ./output_images/fig1.png "Distorted and Undistorted"
[image2-0]: ./output_images/fig2-0.png "Select source points"
[image2]: ./output_images/fig2.png "Straight lines warped"
[image3]: ./output_images/fig3.png "Curved lines warped"
[image4]: ./output_images/fig4.png "Saturation channel thresholding"
[image5]: ./output_images/fig5.png "x-wise gradient thresholding"
[image6]: ./output_images/fig6.png "Combined thresholding"

#### Camera Calibration

The code for this step is contained in `s1_calibration.py` file. The computed calibration matrix `mtx.dat` and distortion coefficients `dist.dat` are saved in `./data/` folder.

The camera calibration was done by processing chessboard images taken from different angles. The images were imported from `camera_cal/` by means of `glob` library.
Then, the calibration pipeline consisted of the following steps.

* Prepare a list of _object points_, i.e., the 3D coordinates of the real undistorted chessboard corners. These are the same for each calibration images. Therefore, an array `obj_pt` containing the coordinates for one image was generated once and then used to populate the list `obj_pts`. (L 44:64)

* The coordinates of the chessboard corners in the 2D image, the _image points_, were automatically extracted from each image by means of `cv2.findChessboardCorners()` function (L 61). These corners were appended to `img_pts` list. The `cv2` function takes grayscale images as input, hence all the calibration images were first converted (`preprocessImage()`).

* `img_pts` and `obj_pts` were used to compute the camera calibration matrix and the distortion coefficients through `cv2.calibrateCamera()`. The images can also be visualised in the undistorted form by applying `undistortImage()`. (L 67, 75)

_Distorted with corners extracted, and undistorted form of the first calibration image_

![alt text][image1]

#### Perspective transform

The code for my perspective transform is in `s2_perspective_transform.py`. It includes the function `warpImage()` (L 9). This function takes as input the image name as a string and the `mtx`, `dist`, and `M` parameters. The first two, can be loaded from `./data/` as they were computed in the previous step. `M` is the perspective transform matrix. This was computed in the following steps:

* Compute the undistorted form of the image with functions from `s1_calibration.py` (L19:25).

* Select by hand the source points `P1`, `P2`, `P3`, `P4` and store them in the array `src_pts`. This was done on pyplot interactive window (L 27:32)
![alt text][image2-0]

* Define the destination points `dst_pts` in function of the image size (L 34:40).

* Get the perspective transform matrix `M` by means of `cv2.getPerspectiveTransform()` and save `M` in `./data/M.dat` (L 42:47).

* Warp image to obtain the bird's-eye view

  _(left) Undistorted test image with straight lines. The source points are marked. (right) Warped image; the destination points are indicated by the dashed line. The lane lines are parallel to the dashed lines._

  ![alt text][image2]

* Apply the pipeline to an image with curved lines

  _Original curved lines image and warped version. The lane lines are not parallel to the dashed lines in the bird's-eye view._

  ![alt text][image3]

#### Thresholding

Two thresholding methods were employed (see `s3_thresholding.py`):

* __Color thresholding__
  * The saturation channel was extracted from the RGB image. This was observed to be the most robust channel upon which perform the color thresholding.
  * The S channel image was warped through `warpImage()` and thresholded by means of two values stored in the tuple `S_thresholds`. These values were selected by trial/error on the test image.
  
  ![alt text][image4]


* __Gradient thresholding__
  * Convert the image to grayscale and warp to fix the perspective.
  * Compute x-wise gradient by means of `cv2.Sobel()`.
  * Take the absolute values and scale the gradient image.
  * Apply Sobel thresholds.

  ![alt text][image5]

The combined thresholding was obtained by operating the _union_ of the two binary images:
```python
mix_bin[(color_th == 1) | (sobel_th == 1)] = 1
```

_Combined thresholding pipeline. The resulting image is on the bottom right._

![alt text][image6]














####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
