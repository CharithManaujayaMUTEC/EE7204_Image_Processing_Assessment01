import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load Host Image
host_path = "Q3/Image_3.jpg"
host_img = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)
if host_img is None:
    raise FileNotFoundError("Host image not found.")

host_img = np.float32(host_img)

# Load Watermark Image
watermark_path = "Q7/watermark.png" 
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
if watermark is None:
    raise FileNotFoundError("Watermark image not found.")

# Resize watermark to fit host's top-left quarter
wm_resized = cv2.resize(watermark, (host_img.shape[1]//2, host_img.shape[0]//2))
wm_resized = np.float32(wm_resized) / 255  # normalize to 0-1

# DWT of Host Image
coeffs2 = pywt.dwt2(host_img, 'haar')
cA, (cH, cV, cD) = coeffs2

# Embed Watermark in Approximation Coefficients
alpha = 0.1  # scaling factor
cA_w = cA.copy()
cA_w[:wm_resized.shape[0], :wm_resized.shape[1]] += alpha * wm_resized

# Reconstruct Watermarked Image
watermarked_img = pywt.idwt2((cA_w, (cH, cV, cD)), 'haar')
watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)

# Extract Watermark
coeffs2_w = pywt.dwt2(np.float32(watermarked_img), 'haar')
cA_w_extracted, _ = coeffs2_w
extracted_wm = (cA_w_extracted[:wm_resized.shape[0], :wm_resized.shape[1]] - cA[:wm_resized.shape[0], :wm_resized.shape[1]]) / alpha
extracted_wm = np.clip(extracted_wm, 0, 1)

# Display Results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(host_img, cmap='gray')
plt.title("Original Host")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(watermarked_img, cmap='gray')
plt.title("Watermarked Image")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(extracted_wm, cmap='gray')
plt.title("Extracted Watermark")
plt.axis('off')

plt.tight_layout()
plt.savefig("Q7/watermark_output.png")
plt.show()