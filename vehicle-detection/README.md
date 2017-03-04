# Vehicle Detection

#### Objectives

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/figure01.png
[image2]: ./output_images/figure02.png
[image3]: ./output_images/figure03.png
[image4]: ./output_images/figure04.png
[image5]: ./output_images/figure05.png

---

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in `vd1_hog.py`.

First, I imported _vehicle_ and _non vehicle_ images (L 61:65).

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) by means of the code from the lectures (L 84:110).  

Here is an example using `GRAY` color space and different combinations of `orientations` and `pixels_per_cell`, whereas `cells_per_block` was set constant to `2`.

![alt text][image2]

### Classification step

The code for this step is contained in `vd_train_clf.py`.

After some tweaking, I settled for the following parameters
```python
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True
```

I also rewrote the lessons' features extraction functions in `vdLib.py`. The function `extract_features()` was used to process all the `cars` and `notcars` images (L 36:46). The features were scaled with `sklearn.preprocessing.StandardScaler()` and split in training and testing datasets (L 47:57).

A simple linear support vector machine classifier `LinearSVC()` was trained and its accuracy was tested on the test dataset (L 59:70). The classifier achieved an accuracy of the _98.9%_.

All the image processing parameters and the trained classifier were saved in a pickle file, to be used in the following steps (L 71:81).

### Sliding Window Search

The code for this step is contained in `main.py`.

The sliding window search function was taken from lecture notes and adapted to handle lists of multiple window settings. The image search is controlled by means of the class `CarScanner` in `vdLib.py`, where the `scan()` method identify the windows coordinates that are passed to the `search_window()` function which extract the features and operates the classification step.

Multiple window sizes were chosen to cover the frame with different cell density depending on the image area. The idea was to use small cells for the upper center of the road, where cars far away appear smaller. Then, while increasing the starting `y` coordinate, the window size was increased. In total, three window sets were employed (red, blue, and green in the picture below), with the following parameters

```python
# windows
x_start_stops = [[None,None],  # red
                 [None, None], # green
                 [270,1000]]   # blue

y_start_stops = [[420,650],
                 [400,575],
                 [375,500]]

xy_windows = [(240,150),
              (120,96),
              (60,48)]

xy_overlaps = [(0.75, 0.55),
               (0.75, 0.5),
               (0.75, 0.5)]
```

![alt text][image3]

The windows were made longer along _x_ than along _y_ because cars looks like horizontal rectangles. Also, the windows were overlapped more in the _x_-wise direction to better capture horizontal features.

In the following picture, the detection pipeline was run over the test images. All the windows were merged in a single window by means of heat map thresholding. The pipeline detects some false positive that can be eliminated by averaging consecutive frames (for bot the averaging and the heat map thresholding, see the following sections).

![alt text][image4]

---

### Video implementation

The processed video can be watched here.

All the windows containing a car (or part of it) were recorded within each video frame. These windows were used to create a heat map (`add_heat()`) that could be thresholded (`apply_threshold()`) to select only the locations were multiple windows were indicating (`scipy.ndimage.measurements.label()`).

The windows in a single frame, resulting from the heat map thresholding, were added to a _frames stack_ which contains the windows from previous frames (`add_frames`). The stack was then summed and the resulting image was thresholded to eliminate false positives. The stack had a maximum length (`max_queue_len=25`) and the heat map threshold was set equal to `25`. Therefore, only the pixels bounded by at least _25_ windows were classified as belonging to a car.

Here is an example fram with all the windows found and the resulting heat map. By setting a threshold of _15_, the false positive on the left can be removed.

![alt text][image5]

---

### Discussion

The main problem with this implementation is the whooping processing time of _2.23s/it_ (yes, seconds per iteration, not the other way around). This is mainly due to the large number (more than _300_) of windows to be searched in each frame. By neglecting the two denser layers the processing can be done at _1.7it/s_ (but with a huge increase in false positives). A major increase in performance should be achievable by exploiting GPUs and rewriting the pipeline to exploit multiple threads. Eventually, a deep learning approach can be employed (as already done by [others](https://github.com/ksakmann/CarND-Vehicle-Detection#comparsion-to-yolo)) while avoiding to explicitly define a computer vision pipeline.
