import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
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

# load parameters
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
cell_per_block = dist_pickle["cell_per_block"]
pix_per_cell = dist_pickle["pix_per_cell"]
spatial_size = dist_pickle["spatial_size"]
spatial_feat = dist_pickle["spatial_feat"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
hist_bins = dist_pickle["hist_bins"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
svc = dist_pickle["svc"]

# windows
x_start_stops = [
				[None,None],
				[None, None],
				[270,1000]
				]

y_start_stops = [
				[420,650],
				[400,575],
				[375,500]
				]

xy_windows = [
				(240,150),
				(120,96),
				(60,48)
			]

xy_overlaps = [
				(0.75, 0.55),
				(0.75, 0.5),
				(0.75, 0.5)
			]

# initialise car scanner istance
scanner = vdLib.CarScanner(color_space=color_space, orient=orient,
			pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
			hog_channel=hog_channel, spatial_size=spatial_size,
			hist_bins=hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat,
			hog_feat=hog_feat, y_start_stops=y_start_stops,
			x_start_stops=x_start_stops, xy_windows=xy_windows,
			xy_overlaps=xy_overlaps, heat_threshold=30, X_scaler=X_scaler,
			clf=svc, max_queue_len=25)

# process video
filename = "project"
input_video = "./{}_video.mp4".format(filename)
output_video = "./{}_out.mp4".format(filename)

clip = VideoFileClip(input_video)
out_clip = clip.fl_image(scanner.scan)
out_clip.write_videofile(output_video, audio=False)

# generate images for the readme
process_images = False
if process_images:
	# draw windows on test image
	test_img = "test_images/test4.jpg"
	dv_img, windows = scanner.scan(test_img)

	sns.set_style("white")
	fig = plt.figure(1, figsize=(6,3))
	fig.clf()
	ax = fig.add_subplot(111)

	i = 0
	for w in windows:
		if i < 40:
			c = (255,0,0)
			i += 1
		elif i < 122:
			c = (0,255,0)
			i+=1
		else:
			c = (0,0,255)
		cv2.rectangle(dv_img, w[0], w[1], c, 4)
	ax.imshow(dv_img)
	ax.set_xlabel("$x$")
	ax.set_ylabel("$y$")
	plt.tight_layout()
	plt.show()
	plt.savefig("output_images/figure03.png")

	# search test images
	sns.set_style("white")
	fig = plt.figure(3, figsize=(12,9))
	fig.clf()
	test_imgs = glob.glob("test_images/*.jpg")
	i = 1
	for img in test_imgs:
		v_img = scanner.scan(img)
		ax = fig.add_subplot(3,2,i)
		ax.imshow(v_img)

		if i in [1,3,5]:
			ax.set_ylabel("$y$")
		if i in [5, 6]:
			ax.set_xlabel("$x$")
		i += 1 
	plt.tight_layout()
	plt.show()
	plt.savefig("output_images/figure04.png")

	# plot heat map
	dv_img, hm = scanner.scan(test_img)

	sns.set_style("white")
	fig = plt.figure(4, figsize=(9,3))
	fig.clf()
	ax = fig.add_subplot(121)
    
	ax.imshow(dv_img)
	ax.set_title("Camera image")

	ax = fig.add_subplot(122)
    
	sns.heatmap(np.sum(hm,2), xticklabels=False, yticklabels=False,
		cbar=1, ax=ax, cmap=sns.cubehelix_palette(8, start=2, rot=0,
						dark=0, light=.95, as_cmap=True))
	ax.set_xlabel("$x$")
	ax.set_ylabel("$y$")
	
	ax.set_title("Heat map")
	plt.tight_layout()
	plt.show()
	plt.savefig("output_images/figure05.png")
