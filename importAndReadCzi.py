import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import czifile
#from aicsimageio import AICSImage
#from czifile import czi2tif

#Read .czi and select colour channel
image = czifile.imread('H99_48hrs-04.czi')
cells = image[0,0,:,:,:]

plt.imshow(cells, cmap = 'gray')

#Threshold image to binary using OTSU, threshold pixels set to 255
ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap = 'gray')

#Remove noise, opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

from skimage.segmentation import clear_border
opening = clear_border(opening)
plt.imshow(opening, cmap = 'gray')

#Sure background
sure_bg = cv2.dilate(opening, kernel, iterations = 10)
plt.imshow(sure_bg, cmap = 'gray')

#dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#plt.imshow(dist_transform, cmap = 'gray')