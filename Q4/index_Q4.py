import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image 4
image_path = "Q4/Image_4.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check filename.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Build Gaussian Pyramid
gaussian_pyramid = [image]
levels = 5

for i in range(levels):
    image = cv2.pyrDown(image)
    gaussian_pyramid.append(image)

# Display Gaussian Pyramid
plt.figure(figsize=(15,5))
for i in range(len(gaussian_pyramid)):
    plt.subplot(1, len(gaussian_pyramid), i+1)
    plt.imshow(gaussian_pyramid[i])
    plt.title(f"G{i}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q4/gaussian_pyramid.png")
plt.show()

# Build Laplacian Pyramid
laplacian_pyramid = []

for i in range(levels, 0, -1):
    expanded = cv2.pyrUp(gaussian_pyramid[i])
    expanded = cv2.resize(expanded,
                          (gaussian_pyramid[i-1].shape[1],
                           gaussian_pyramid[i-1].shape[0]))
    laplacian = cv2.subtract(gaussian_pyramid[i-1], expanded)
    laplacian_pyramid.append(laplacian)

# Display Laplacian Pyramid
plt.figure(figsize=(15,5))
for i in range(len(laplacian_pyramid)):
    plt.subplot(1, len(laplacian_pyramid), i+1)
    plt.imshow(laplacian_pyramid[i])
    plt.title(f"L{i}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q4/laplacian_pyramid.png")
plt.show()