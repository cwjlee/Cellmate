#conda install fuzzy-c-means

import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import cv2

n_samples = 3000

X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2)),
    np.random.normal((9, 0), size=(n_samples, 2))
))


fcm = FCM(n_clusters=4)
fcm.fit(X)

fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

# plot result
# f, axes = plt.subplots(1, 2, figsize=(11,5))
# axes[0].scatter(X[:,0], X[:,1], alpha=.1)
# axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
# axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
# plt.show()

im_gray = cv2.imread('H99_48hrs-04.czi.tif', cv2.IMREAD_GRAYSCALE)
#im_gray = test[:,:,0]
plt.imshow(im_gray, cmap='gray')
plt.show()

fcm = FCM(n_clusters=3)
fcm.fit(im_gray)
print("fcm fit")

fcm_centers = fcm.centers
fcm_labels = fcm.predict(im_gray)
print("fcm predict")

#fcm_centers = np.uint8(fcm_centers) 
segmented_data = fcm_centers[fcm_labels] 
print("segment")

segmented_image = segmented_data.reshape((im_gray.shape)) 
#plt.savefig('basic-clustering-output.jpg')
plt.imshow(segmented_image)
plt.show()


