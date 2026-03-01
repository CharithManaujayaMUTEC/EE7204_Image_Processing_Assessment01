import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image 3
image_path = "Q3/Image_3.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the file name.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# PART 1: Gaussian Filtering with Different Kernel Sizes
kernel_sizes = [3, 5, 11, 15]
filtered_images = []

for k in kernel_sizes:
    blurred = cv2.GaussianBlur(image_rgb, (k, k), sigmaX=0)
    filtered_images.append(blurred)

plt.figure(figsize=(15,6))

plt.subplot(1,5,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

for i, k in enumerate(kernel_sizes):
    plt.subplot(1,5,i+2)
    plt.imshow(filtered_images[i])
    plt.title(f"{k}x{k}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q3/output_kernel.png")
plt.show()

# PART 2: Effect of Sigma
sigma_values = [1, 5, 10]
kernel_size = 11

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

for i, s in enumerate(sigma_values):
    blurred_sigma = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigmaX=s)
    plt.subplot(1,4,i+2)
    plt.imshow(blurred_sigma)
    plt.title(f"Sigma={s}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q3/output_sigma.png")
plt.show()