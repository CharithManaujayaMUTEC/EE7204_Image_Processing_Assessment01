import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.util import random_noise

# Load Image 3
image_path = "Q6/Image_6.jpg" 
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add Salt & Pepper Noise
sp_noise = random_noise(gray, mode='s&p', amount=0.1)
sp_noise = (sp_noise * 255).astype(np.uint8)

# Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Combine I + SP + L(I)
combined = cv2.add(gray, sp_noise)
combined = cv2.add(combined, laplacian)

# Wavelet Decomposition
# Use Haar
coeffs2 = pywt.dwt2(combined, 'haar')
cA, (cH, cV, cD) = coeffs2

# Remove high-frequency components
cH.fill(0)
cV.fill(0)
cD.fill(0)

# Reconstruct image
reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

# Display Results
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(sp_noise, cmap='gray')
plt.title("Salt & Pepper")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(combined, cmap='gray')
plt.title("I + SP + Laplacian")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(reconstructed, cmap='gray')
plt.title("Wavelet Smooth")
plt.axis('off')

plt.tight_layout()
plt.savefig("Q6/wavelet_output.png")
plt.show() 