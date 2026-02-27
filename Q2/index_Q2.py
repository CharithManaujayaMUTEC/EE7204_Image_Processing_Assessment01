import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# -----------------------------
# Load Image 2
# -----------------------------
image_path = "Q1/image1.png"   # Change if needed
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the file name.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# Add Salt & Pepper Noise
# -----------------------------
noise_10 = random_noise(image_rgb, mode='s&p', amount=0.10)
noise_20 = random_noise(image_rgb, mode='s&p', amount=0.20)

# Convert back to uint8
noise_10 = (noise_10 * 255).astype(np.uint8)
noise_20 = (noise_20 * 255).astype(np.uint8)

# -----------------------------
# Display Original and Noisy Images
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noise_10)
plt.title("10% S&P Noise")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(noise_20)
plt.title("20% S&P Noise")
plt.axis("off")

plt.tight_layout()
plt.savefig("Q2/output_noise.png")
plt.show()

# -----------------------------
# Apply Median Filtering
# -----------------------------
kernel_sizes = [3, 5, 11]

# For 10% Noise
plt.figure(figsize=(12,4))
plt.subplot(1,4,1)
plt.imshow(noise_10)
plt.title("10% Noisy")
plt.axis("off")

for i, k in enumerate(kernel_sizes):
    median_filtered = cv2.medianBlur(noise_10, k)
    plt.subplot(1,4,i+2)
    plt.imshow(median_filtered)
    plt.title(f"{k}x{k}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q2/output_median_10.png")
plt.show()

# For 20% Noise
plt.figure(figsize=(12,4))
plt.subplot(1,4,1)
plt.imshow(noise_20)
plt.title("20% Noisy")
plt.axis("off")

for i, k in enumerate(kernel_sizes):
    median_filtered = cv2.medianBlur(noise_20, k)
    plt.subplot(1,4,i+2)
    plt.imshow(median_filtered)
    plt.title(f"{k}x{k}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q2/output_median_20.png")
plt.show()