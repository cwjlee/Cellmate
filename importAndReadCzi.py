import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import czifile
#from aicsimageio import AICSImage
#from czifile import czi2tif


image = czifile.imread('H99_48hrs-04.czi')

cells = image[0,0,:,:,:]

plt.imshow(cells, cmap = 'gray')