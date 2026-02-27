import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Image 6
# -----------------------------
image_name = "Q6/Image_6.jpg"
img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found.")

# -----------------------------
# Convert to Binary
# -----------------------------
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)

# -----------------------------
# Morphological Operations
# -----------------------------
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# Extract Area & Perimeter
# -----------------------------
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

areas = []
perimeters = []

for cnt in contours:
    areas.append(cv2.contourArea(cnt))
    perimeters.append(cv2.arcLength(cnt, True))

print("Areas:", areas)
print("Perimeters:", perimeters)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12,6))

titles = ["Original", "Binary", "Erosion", "Dilation", "Opening", "Closing"]
images = [img, binary, erosion, dilation, opening, closing]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("Q10/morphological_results.png")
plt.show()