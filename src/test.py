import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('resnet152.png')
image2 = cv2.imread('vit_b_16.png')

# Ensure the images have the same shape
if image1.shape != image2.shape:
    raise ValueError("Images do not have the same dimensions")

# Calculate the per-pixel average
average_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

# Save the resulting image or display it
cv2.imwrite('average_image.png', average_image)
# Or use cv2.imshow to display
# cv2.imshow('Average Image', average_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()