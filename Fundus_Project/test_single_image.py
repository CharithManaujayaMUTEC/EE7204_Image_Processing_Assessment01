import cv2
import numpy as np
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
image_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/images/1_A.png"
mask_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/masks/1_A.png"

results_dir = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/results"
os.makedirs(results_dir, exist_ok=True)

# --------------------------------------------------
# Load Images
# --------------------------------------------------
img = cv2.imread(image_path)
gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# --------------------------------------------------
# Step 1: Green Channel
# --------------------------------------------------
green = img[:, :, 1]

# --------------------------------------------------
# Step 2: Large Gaussian Blur (Better Background Estimation)
# --------------------------------------------------
blur_large = cv2.GaussianBlur(green, (41,41), 0)

# --------------------------------------------------
# Step 3: Illumination Correction
# --------------------------------------------------
corrected = cv2.subtract(blur_large, green)

# --------------------------------------------------
# Step 4: Normalize
# --------------------------------------------------
corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

# --------------------------------------------------
# Step 5: Circular Retina Mask
# --------------------------------------------------
h, w = green.shape
mask_circle = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask_circle, (w//2, h//2), min(h,w)//2 - 10, 255, -1)

corrected = cv2.bitwise_and(corrected, corrected, mask=mask_circle)

# --------------------------------------------------
# Step 6: Manual Threshold (Tune 15–25)
# --------------------------------------------------
threshold_value = 15
_, binary = cv2.threshold(corrected, threshold_value, 255, cv2.THRESH_BINARY)

# --------------------------------------------------
# Step 7: Morphological Cleaning
# --------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# --------------------------------------------------
# Step 8: Remove Small Components
# --------------------------------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)

min_area = 120
cleaned = np.zeros_like(closing)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        cleaned[labels == i] = 255

segmented = cleaned

# --------------------------------------------------
# Dice & Jaccard
# --------------------------------------------------
seg = segmented > 0
gt  = gt_mask > 0

intersection = np.logical_and(seg, gt)
union = np.logical_or(seg, gt)

dice = 2 * intersection.sum() / (seg.sum() + gt.sum() + 1e-8)
jaccard = intersection.sum() / (union.sum() + 1e-8)

print("Dice:", dice)
print("Jaccard:", jaccard)