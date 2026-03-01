import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image 1
image_path = "Q1/Image_1.jpg" 
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the file name.")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Average Filtering
kernel_sizes = [3, 5, 11, 15]
filtered_images = []

for k in kernel_sizes:
    blurred = cv2.blur(image_rgb, (k, k))
    filtered_images.append(blurred)

# Display Results
plt.figure(figsize=(15, 6))

# Original Image
plt.subplot(1, 5, 1)
plt.imshow(image_rgb)
plt.title(f"Original\n{image_rgb.shape}")
plt.axis("off")

# Filtered Images
for i, k in enumerate(kernel_sizes):
    plt.subplot(1, 5, i + 2)
    plt.imshow(filtered_images[i])
    plt.title(f"{k} x {k}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q1/output_Q1.png")
plt.show()