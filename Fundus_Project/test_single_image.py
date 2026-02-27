import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
image_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/images/1_A.png"
mask_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/masks/1_A.png"

# -----------------------------
# Load Images
# -----------------------------
img = cv2.imread(image_path)
gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is None or gt_mask is None:
    raise FileNotFoundError("Image or mask not found.")

# -----------------------------
# Step 1: Extract Green Channel
# -----------------------------
green = img[:, :, 1]

# -----------------------------
# Step 2: Histogram Equalization
# -----------------------------
equalized = cv2.equalizeHist(green)

# -----------------------------
# Step 3: Gaussian Blur
# -----------------------------
blur = cv2.GaussianBlur(equalized, (5,5), 0)

# -----------------------------
# Step 4: High-Pass Filtering
# -----------------------------
high_pass = cv2.subtract(equalized, blur)

# -----------------------------
# Step 5: Marr–Hildreth (LoG)
# -----------------------------
log = cv2.Laplacian(high_pass, cv2.CV_64F)
log = cv2.convertScaleAbs(log)

# -----------------------------
# Step 6: Invert (vessels dark originally)
# -----------------------------
inverted = cv2.bitwise_not(log)

# -----------------------------
# Step 7: Global Threshold
# -----------------------------
_, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)

# -----------------------------
# Step 8: Morphological Opening
# -----------------------------
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# -----------------------------
# Step 9: Morphological Closing
# -----------------------------
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# Remove Small Components
# -----------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)

min_area = 50
cleaned = np.zeros_like(closing)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        cleaned[labels == i] = 255

segmented = cleaned

# -----------------------------
# Convert masks to binary logical
# -----------------------------
seg = segmented > 0
gt  = gt_mask > 0

# -----------------------------
# Dice & Jaccard
# -----------------------------
intersection = np.logical_and(seg, gt)
union = np.logical_or(seg, gt)

dice = 2 * intersection.sum() / (seg.sum() + gt.sum() + 1e-8)
jaccard = intersection.sum() / (union.sum() + 1e-8)

print("Dice:", dice)
print("Jaccard:", jaccard)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(green, cmap='gray')
plt.title("Green Channel")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(high_pass, cmap='gray')
plt.title("High Pass")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(log, cmap='gray')
plt.title("LoG")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(segmented, cmap='gray')
plt.title("Segmented")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()