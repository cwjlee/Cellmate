import matplotlib.pyplot as plt
import cv2

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

img = cv2.imread('./Images/CL1.czi.tif')

# Compute a mask
lum = color.rgb2gray(img)
maskCell = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum < 0.75, 500),
    500)
maskCap = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum < 0.5, 500),
    500)

maskCell = morphology.opening(maskCell, morphology.disk(3))
plt.imshow(maskCell)
plt.show()
maskCap = morphology.opening(maskCap, morphology.disk(3))
plt.imshow(maskCap)
plt.show()

# # SLIC result
# slic = segmentation.slic(img, n_segments=200, start_label=1)

# # maskSLIC result
# m_slic = segmentation.slic(img, n_segments=100, mask=mask, start_label=1)

# # Display result
# fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
# ax1, ax2, ax3, ax4 = ax_arr.ravel()

# ax1.imshow(img)
# ax1.set_title('Original image')

# ax2.imshow(mask, cmap='gray')
# ax2.set_title('Mask')

# ax3.imshow(segmentation.mark_boundaries(img, slic))
# ax3.contour(mask, colors='red', linewidths=1)
# ax3.set_title('SLIC')

# ax4.imshow(segmentation.mark_boundaries(img, m_slic))
# ax4.contour(mask, colors='red', linewidths=1)
# ax4.set_title('maskSLIC')

# for ax in ax_arr.ravel():
#     ax.set_axis_off()

# plt.tight_layout()
# plt.show()