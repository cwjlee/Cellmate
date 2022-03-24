#conda install fuzzy-c-means

import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import cv2
from PIL import Image


image = cv2.imread('H99_48hrs-04.czi.tif')
plt.imshow(image, cmap='gray')
N, M = 2048

X = (
    np.asarray(image)                              
    .reshape((N*M, 3))                             
)


fcm = FCM(n_clusters=3)
fcm.fit(X)

fcm_labels = fcm.predict(X)
transformed_X = fcm.centers[fcm_labels]

quatized_array = (
    transformed_X
    .astype('uint8')                               
    .reshape((M, N, 3))                            
)

quatized_image = Image.fromarray(np.asarray(quatized_array))
plt.imshow(quatized_image)




