#conda install fuzzy-c-means

import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from czifile import czi2tif


#czi2tif('./Images/CL4.czi')
image = cv2.imread('./Images/CL1.czi.tif')

plt.imshow(image, cmap='gray')
plt.show()
N, M = 2048, 2048

X = np.asarray(image)
#X = np.clip(X, None, 90)
X = X.reshape((N*M,3))
'''Y = (
    X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)
array_image = Image.fromarray(np.asarray(Y))
plt.imshow(array_image)'''

'''X = (
    np.asarray(image)                              # convert a PIL image to np array
    .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
)'''


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






