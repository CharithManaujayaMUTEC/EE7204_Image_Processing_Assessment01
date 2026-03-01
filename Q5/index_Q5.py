import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# Load Image 1 (Same as Q1)
# ----------------------------------
image_path = "Q1/Image_1.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ----------------------------------
# Parameters
# ----------------------------------
kernel_size = 15
pad = kernel_size // 2

# ----------------------------------
# Manual Average Filtering (A)
# ----------------------------------
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Use convolution manually
manual_filtered = cv2.filter2D(image_rgb, -1, kernel)

# ----------------------------------
# Built-in OpenCV Average Filtering (B)
# ----------------------------------
builtin_filtered = cv2.blur(image_rgb, (kernel_size, kernel_size))

# ----------------------------------
# Difference (A - B)
# ----------------------------------
difference = cv2.absdiff(manual_filtered, builtin_filtered)

# ----------------------------------
# Display Results
# ----------------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(manual_filtered)
plt.title("Manual (A)")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(builtin_filtered)
plt.title("Built-in (B)")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(difference)
plt.title("Difference (A-B)")
plt.axis("off")

plt.tight_layout()
plt.savefig("Q5/output_Q5.png")
plt.show()