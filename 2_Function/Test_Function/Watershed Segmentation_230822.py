import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

# # Image loading
img = cv2.imread("./174101_Origin.jpg", cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]
print(width,height)
cut=int(width*0.005)
img = img[:,cut : width-cut]
# img = cv2.add(img,10)
# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bin_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Image grayscale conversion and contrast adjustment
bin_img = np.clip((1 + 2.3) * bin_img - 128 * 2.3, 0, 255).astype(np.uint8)
threshold_value = 100  # 임계값 설정
ret, thresholded_img = cv2.threshold(bin_img, threshold_value, 255, cv2.THRESH_BINARY)

ret, binary_img = cv2.threshold(thresholded_img,  245 , 255, cv2.THRESH_BINARY_INV)
inverted_binary_img = cv2.bitwise_not(binary_img)

# Create subplots using GridSpec
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(1, 3, wspace=0.1, hspace=0.1)

# Plot original image
ax1 = plt.subplot(grid[0, 0])
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("origin Image")
ax1.axis('off')

# Plot cv2.dilate image
ax2 = plt.subplot(grid[0, 1])
ax2.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))
ax2.set_title("cv2.dilate Image")
ax2.axis('off')

# Plot binary image
ax3 = plt.subplot(grid[0, 2])
ax3.imshow(inverted_binary_img, cmap='gray')
ax3.set_title("binary Image")
ax3.axis('off')

plt.tight_layout()
plt.show()
