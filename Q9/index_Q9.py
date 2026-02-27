import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load MRI Image
# -----------------------------
image_name = "Q5/Image_5.jpg"   # clearly state image name
mri = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

if mri is None:
    raise FileNotFoundError("MRI image not found.")

# -----------------------------
# 1. Histogram Equalization
# -----------------------------
hist_eq = cv2.equalizeHist(mri)

# -----------------------------
# 2. Contrast Stretching
# -----------------------------
min_val = np.min(mri)
max_val = np.max(mri)

contrast_stretch = ((mri - min_val) / (max_val - min_val)) * 255
contrast_stretch = contrast_stretch.astype(np.uint8)

# -----------------------------
# 3. Gaussian Noise Reduction
# -----------------------------
gaussian = cv2.GaussianBlur(mri, (5,5), 0)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(mri, cmap='gray')
plt.title("Original MRI")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(hist_eq, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(contrast_stretch, cmap='gray')
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Filtering")
plt.axis("off")

plt.tight_layout()
plt.savefig("Q9/mri_enhancement.png")
plt.show()