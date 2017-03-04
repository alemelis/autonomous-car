import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import time
from moviepy.editor import VideoFileClip
import pickle
import glob

import vdLib
import importlib
importlib.reload(vdLib)

# import images
car_imgs = glob.glob("vehicles/**/*.png")
notcar_imgs = glob.glob("non-vehicles/**/*.png")

car_img = vdLib.read_image(car_imgs[0])
notcar_img = vdLib.read_image(notcar_imgs[0])

# parameters
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

# extract features
car_features = vdLib.extract_features(car_imgs, color_space, orient,
                                      spatial_size, hist_bins, pix_per_cell,
                                      cell_per_block, spatial_feat, hist_feat,
                                      hog_feat, hog_channel)

notcar_features = vdLib.extract_features(notcar_imgs, color_space, orient,
                                      spatial_size, hist_bins, pix_per_cell,
                                      cell_per_block, spatial_feat, hist_feat,
                                      hog_feat, hog_channel)

# split training and test set
X_features = np.vstack((car_features, notcar_features)).astype(np.float64)
y_features = np.hstack((np.ones(len(car_imgs)), np.zeros(len(notcar_imgs))))

X_train, X_test, y_train, y_test = train_test_split(X_features, y_features,
                                             test_size=0.2, random_state=42)
# scale features
X_scaler = StandardScaler().fit(X_features)

X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

# train classifier
svc = LinearSVC()
svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])

# save parameters
params = {'color_space': color_space, 'orient': orient,
            'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
            'hog_channel': hog_channel, 'spatial_size': spatial_size,
            'hist_bins': hist_bins, 'spatial_feat': spatial_feat,
            'hist_feat': hist_feat, 'hog_feat': hog_feat,
            'hog_channel': hog_channel, 'svc': svc, 'X_scaler': X_scaler}

with open('svc_pickle.p', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
