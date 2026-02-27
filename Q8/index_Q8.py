import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load CT Image
# -----------------------------
ct_path = "Q4/Image_4.jpg"  # Change filename if needed
ct_img = cv2.imread(ct_path)

if ct_img is None:
    raise FileNotFoundError("CT image not found.")

gray = cv2.cvtColor(ct_img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Step 1: Gaussian Smoothing
# -----------------------------
blur = cv2.GaussianBlur(gray, (5,5), 0)

# -----------------------------
# Step 2: Thresholding
# -----------------------------
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# -----------------------------
# Step 3: Morphological Operations
# -----------------------------
kernel = np.ones((5,5), np.uint8)

closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# -----------------------------
# Step 4: Extract Segmented Organ
# -----------------------------
segmented = cv2.bitwise_and(gray, gray, mask=opening)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Original CT")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(opening, cmap='gray')
plt.title("After Morphology")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(segmented, cmap='gray')
plt.title("Segmented Organ")
plt.axis('off')

plt.tight_layout()
plt.savefig("Q8/segmentation_output.png")
plt.show()