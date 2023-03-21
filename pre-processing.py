import numpy as np
from matplotlib import pyplot as plt
import cv2
import skimage.io
import skimage.filters
from fcmeans import FCM
from PIL import Image


example_image = cv2.imread('H99_48hrs-04.czi.tif')
plt.imshow(example_image)
plt.show()
N, M = 2048, 2048

sigma = 4.0
blurred = skimage.filters.gaussian(
    example_image, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
plt.imshow(blurred)
plt.show()

X = (
    np.asarray(blurred)                              # convert a PIL image to np array
    .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
)


fcm = FCM(n_clusters=3)
fcm.fit(X)

fcm_labels = fcm.predict(X)
transformed_X = fcm.centers[fcm_labels]

quantized_array = (
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)

quantized_image = Image.fromarray(np.asarray(quantized_array))
plt.imshow(quantized_image)
plt.show()
