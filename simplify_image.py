# cluster an image into sections which account for color and location
# turn these sections into outlines

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
from fourier_series import Fourier_Series


def simplify_image(image_file, COLOR_WEIGHT=0.5, CLUSTERS=100, BLUR_LEVEL=5, show_steps=True):
	# load image
	im_orig = cv.imread(image_file)
	im_rgb = cv.cvtColor(im_orig, cv.COLOR_BGR2RGB)
	if show_steps:
		plt.imshow(im_rgb)
		plt.title("Original Image")
		plt.show()

	# blur image
	im = cv.GaussianBlur(im_rgb, (BLUR_LEVEL,BLUR_LEVEL),0)

	if show_steps:
		plt.imshow(im)
		plt.title("Blurred Image")
		plt.show()

	row =np.repeat(np.expand_dims(np.arange(im.shape[0]),axis=1), im.shape[1], axis=1)
	col = np.repeat(np.expand_dims(np.arange(im.shape[1]),axis=0), im.shape[0], axis=0)
	ind_arr = np.dstack((row,col)).reshape(im.shape[0]*im.shape[1],2)

	pixels_list = np.hstack((ind_arr.astype(float),COLOR_WEIGHT*im.reshape(im.shape[0]*im.shape[1],3).astype(float)))


	# cluster image
	sections = scipy.cluster.vq.kmeans2(pixels_list, CLUSTERS)[0]
	pixels_in_sec = [set() for i in range(CLUSTERS)]

	# assign each pixel to a cluster
	for i in ind_arr:
		val = np.hstack((i, COLOR_WEIGHT*im[tuple(i)].astype(float)))
		dist_to_sections = np.linalg.norm(sections-val,axis=1)
		section_ind = np.argmin(dist_to_sections)

		new_color = (sections[section_ind][2:]/COLOR_WEIGHT).astype(np.uint8)
		im[tuple(i)] = new_color

		pixels_in_sec[section_ind].add(tuple(i))

	if show_steps:
		plt.imshow(im)
		plt.title("Clustered Image")
		plt.show()

	# find outlines of clusters
	for i in range(CLUSTERS):
		pixels = pixels_in_sec[i]
		inside = set()
		for j in pixels:
			r,c = j
			if (r+1,c) in pixels and (r-1,c) in pixels and (r,c+1) in pixels and (r,c-1) in pixels:
				inside.add(j)

		pixels_in_sec[i] = pixels - inside

		if show_steps:
			## DELETE LATER
			for j in inside: ####
				im[j] = np.array([255,255,255],dtype=np.uint8) ####


	if show_steps:
		plt.imshow(im)
		plt.title("Cluster Outlines")
		plt.show()


	# returns the cluster centers and which pixels are in each cluster
	return sections*np.array([1,1,1/COLOR_WEIGHT,1/COLOR_WEIGHT,1/COLOR_WEIGHT]), pixels_in_sec




