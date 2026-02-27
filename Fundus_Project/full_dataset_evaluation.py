import cv2
import numpy as np
import os

# --------------------------------------------------
# FIXED PARAMETERS
# --------------------------------------------------
GAUSSIAN_KERNEL = 41

OPEN_KERNEL_SIZES = [3, 5, 7]
MIN_AREA_VALUES = [80, 120, 160]

# --------------------------------------------------
# Segmentation Function (Improved Morphological Pipeline)
# --------------------------------------------------
def segment_vessels(image, open_size, min_area):

    green = image[:, :, 1]

    # 1️⃣ Illumination correction
    blur_large = cv2.GaussianBlur(green, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    corrected = cv2.subtract(blur_large, green)

    # 2️⃣ Morphological Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    opened = cv2.morphologyEx(corrected, cv2.MORPH_OPEN, kernel_open)

    # 3️⃣ Enhance thin vessels
    enhanced = cv2.subtract(corrected, opened)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # 4️⃣ Circular retina mask
    h, w = green.shape
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, (w//2, h//2), min(h, w)//2 - 10, 255, -1)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask_circle)

    # 5️⃣ Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6️⃣ Closing to reconnect vessels
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 7️⃣ Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    cleaned = np.zeros_like(closed)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------
def evaluate_dataset(images_path, masks_path, open_size, min_area):

    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    dice_scores = []

    for file in image_files:

        img_path = os.path.join(images_path, file)
        mask_path = os.path.join(masks_path, file)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        segmented = segment_vessels(img, open_size, min_area)

        seg = segmented > 0
        gt  = gt_mask > 0

        intersection = np.logical_and(seg, gt)
        dice = 2 * intersection.sum() / (seg.sum() + gt.sum() + 1e-8)

        dice_scores.append(dice)

    return np.mean(dice_scores)


# --------------------------------------------------
# TRAINING SET GRID SEARCH
# --------------------------------------------------
training_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/images"
training_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/masks"

best_dice = 0
best_params = (None, None)

print("Running Morphology Grid Search on Training Set...\n")

for open_size in OPEN_KERNEL_SIZES:
    for min_area in MIN_AREA_VALUES:

        avg_dice = evaluate_dataset(training_images_path,
                                     training_masks_path,
                                     open_size,
                                     min_area)

        print(f"OPEN={open_size}, MIN_AREA={min_area} → Dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_params = (open_size, min_area)

print("\nBest Training Parameters:")
print("OPEN_KERNEL_SIZE:", best_params[0])
print("MIN_AREA:", best_params[1])
print("Best Training Dice:", best_dice)


# --------------------------------------------------
# VALIDATION USING BEST PARAMETERS
# --------------------------------------------------
validation_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/images"
validation_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/masks"

val_dice = evaluate_dataset(validation_images_path,
                            validation_masks_path,
                            best_params[0],
                            best_params[1])

print("\nValidation Dice with Best Parameters:", val_dice)