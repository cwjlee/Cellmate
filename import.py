import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import czifile

image = czifile.imread('H99_48hrs-04.czi')
print(image.shape)
