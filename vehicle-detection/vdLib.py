import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import glob, time
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

def read_image(image_name):
    return mpimg.imread(image_name)

def extract_features(images, cspace='RGB', orient=9, spatial_size=(32, 32),
                     hist_bins=32, pix_per_cell=8, cell_per_block=2,
                     spatial_feat=True, hist_feat=True, hog_feat=True,
                     hog_channel=0):
    features = []
    # Iterate through the list of images
    for image in images:
        file_features = []

        # load image
        if type(image) == str:
            image = read_image(image)

        # convert color space
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            file_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            file_features.append(hist_features)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell,
                                                     cell_per_block, vis=False,
                                                     feature_vec=True))
            hog_features = np.ravel(hog_features)
            file_features.append(hog_features)
        else:
            hog_features=get_hog_features(feature_image[:,:,hog_channel],orient,
                                          pix_per_cell, cell_per_block,
                                          vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return np.array(features)

def get_hog_features(img, orient, pix_per_cell,
                     cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            transform_sqrt=False, visualise=True,
                            feature_vector=False)
        return features.ravel(), hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False, visualise=False,
                       feature_vector=feature_vec)
        return features.ravel()

def bin_spatial(image, size=(32, 32)):

    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()

    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):

    color_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_2_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_3_hist = np.histogram(img[:, :, 0], bins=nbins)

    return np.concatenate((color_1_hist[0], color_2_hist[0], color_3_hist[0]))

class CarScanner:

    def __init__(self, color_space, orient, pix_per_cell, cell_per_block,
                 hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat,
                 hog_feat, y_start_stops, x_start_stops, xy_windows,
                 xy_overlaps, heat_threshold, X_scaler, clf, max_queue_len):

        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.y_start_stops = y_start_stops
        self.x_start_stops = x_start_stops
        self.xy_windows = xy_windows
        self.xy_overlaps = xy_overlaps
        self.heat_threshold = heat_threshold
        self.X_scaler = X_scaler
        self.clf = clf

        self.frames = []
        self.max_queue_len = max_queue_len

        self.first_frame = True
        self.slided_window = []

    def scan(self, input_image):
        if type(input_image) == str:
            input_image = read_image(input_image)

        copy_image = np.copy(input_image).astype(np.float32)/255

        # the window location can be find once at the beginning of the video
        if self.first_frame:
            self.slided_windows = slide_windows(copy_image,
                                x_start_stops=self.x_start_stops,
                                y_start_stops=self.y_start_stops,
                                xy_windows=self.xy_windows,
                                xy_overlaps=self.xy_overlaps)
            self.first_frame = False

        # all the windows are checked in each frame
        on_windows = search_windows(copy_image, self.slided_windows,
                            self.clf, self.X_scaler,
                            color_space=self.color_space,
                            spatial_size=self.spatial_size,
                            hist_bins=self.hist_bins, orient=self.orient,
                            pix_per_cell=self.pix_per_cell,
                            cell_per_block=self.cell_per_block,
                            hog_channel=self.hog_channel,
                            spatial_feat=self.spatial_feat,
                            hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        heat_map = np.zeros_like(copy_image)
        heat_map = add_heat(heat_map, on_windows)

        self.frames.insert(0, heat_map)

        frames_stack = self.add_frames()
        heat_map = apply_threshold(frames_stack, self.heat_threshold)

        labels = label(heat_map)

        image_with_bb = draw_labeled_bboxes(input_image, labels)

        return image_with_bb#, self.slided_windows
        # for w in on_windows:
        #     cv2.rectangle(input_image, w[0], w[1], (255,0,0), 3)
        # return input_image, heat_map

    def add_frames(self):
        if len(self.frames) > self.max_queue_len:
            self.frames.pop()

        frames_stack = np.array(self.frames)

        return np.sum(frames_stack, 0)

def slide_windows(img, x_start_stops=[None, None], y_start_stops=[None, None],
                 xy_windows=(64, 64), xy_overlaps=(0.5, 0.5)):

    windows = []
    for i in range(len(x_start_stops)):
        windows = slide_window(img, x_start_stops[i], y_start_stops[i],
                                xy_windows[i], xy_overlaps[i], windows)
    return windows

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5), window_list=[]):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list

def search_windows(img, windows, clf, X_scaler, color_space, spatial_size,
                    hist_bins, orient, pix_per_cell, cell_per_block,
                    hog_channel, spatial_feat, hist_feat, hog_feat):

    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))

        extracted_features = extract_features([test_img], cspace=color_space,
                                    orient=orient, spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hog_feat, hog_feat=hist_feat,
                                    hog_channel=hog_channel)

        test_features=X_scaler.transform(np.array(extracted_features).reshape(1,
                                                                            -1))
        prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255, 255, 0), 3)

    # Return the image
    return img
