import csv

import os

from PIL import Image
import cv2

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle as sk_shuffle

from keras.models import Sequential
from keras.layers import Lambda, Convolution2D,\
	MaxPooling2D, Dropout, Flatten, Dense, Conv2D

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Parameters
clean_sky = 50 # horizon pixel 
new_rows  = 11 # resized image size
new_cols  = 32
stering_theta = 0.3 # side cameras correction angle
test_samples = 20   # use few images to test the CNN
epochs = 40
batch_size = 128

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

def loadAndPreprocessImage(log, idx, new_cols, new_rows):
	img = Image.open("IMG/%s"%log[idx].split('/')[-1]) # open .jpg
	img = np.array(img)							     # convert to numpy array

	# extract saturation channel and crop out the sky
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[clean_sky:,:,1]  
	return cv2.resize(img, (new_cols, new_rows))     # return resized image

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Load training data

## open log file
print("Load .csv file")
logs = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		logs.append(row)

## preprocess images
print("Load training dataset")
X = []
y = []
counter = 0
for log in logs:
	# central camera
	X.append(loadAndPreprocessImage(log, 0, new_cols, new_rows))
	y.append(float(log[3]))

	# left camera
	X.append(loadAndPreprocessImage(log, 1, new_cols, new_rows))
	y.append(float(log[3]) + stering_theta)

	# right camera
	X.append(loadAndPreprocessImage(log, 2, new_cols, new_rows))
	y.append(float(log[3]) - stering_theta)

	if counter%1000 == 0:
		print("#", sep=' ', end='', flush=True)
	counter += 1 
print("\n")

X = np.array(X)
y = np.array(y)

## flip horizontally the images to augment the dataset
X = np.concatenate([X, X[:,:,::-1]])
y = np.concatenate([y, -y])

## shuffle and split in training and test data
X, y = sk_shuffle(X, y)

X_train = X[:-test_samples,:,:,None]
y_train = y[:-test_samples]
X_test = X[-test_samples:,:,:,None]
y_test = y[-test_samples:]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Build the model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
	input_shape=(new_rows, new_cols, 1), name='Normalization'))
model.add(Conv2D(2, 1, 1, border_mode='same', activation='relu'))
model.add(MaxPooling2D((4, 4), (4, 4), 'same'))
model.add(Dropout(0.3))												
model.add(Flatten())
model.add(Dense(1))  # one neuron to rule them all

model.summary() # print model summary

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Training
model.compile(loss='mean_squared_error',optimizer='adam')

# train the model using the 10% of the dataset for validation
model.fit(X_train, y_train, batch_size=batch_size,
	nb_epoch=epochs, verbose=1, validation_split=0.1)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Save model
print("Save model")
model_json = model.to_json()
import json
with open('model.json', 'w') as f:
	json.dump(model_json, f, ensure_ascii=False)
model.save_weights("model.h5")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

# Test run
P = model.predict(X_test[:])
P = P.reshape((P.shape[0],)) 

## plot predictions over line of equality
sns.set_style("white")
fig = plt.figure(1, figsize=(5,5))
fig.clf()
ax = fig.add_subplot(111)
plt.plot([-0.5,0.5],[-0.5,0.5], 'k--', label="line of equality")
ax.scatter(P, y_test, marker='o', color="orchid", s=70, zorder=10)
plt.xlabel("prediction")
plt.ylabel("y_test")
plt.legend(loc='best')
plt.tight_layout()
plt.draw()
plt.show()