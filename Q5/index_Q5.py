import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Image 5
# -----------------------------
image_path = "Q5/Image_5.jpg"   # Change filename if needed
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check filename.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Sobel Edge Detection
# -----------------------------
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# -----------------------------
# Laplacian Edge Detection
# -----------------------------
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# -----------------------------
# Canny Edge Detection
# -----------------------------
canny = cv2.Canny(gray, 100, 200)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.savefig("Q5/edge_detection.png")
plt.show()