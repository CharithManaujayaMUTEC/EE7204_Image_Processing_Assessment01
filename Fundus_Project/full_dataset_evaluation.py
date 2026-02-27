import cv2
import numpy as np
import os

# --------------------------------------------------
# FINAL FROZEN PARAMETERS
# --------------------------------------------------
GAUSSIAN_KERNEL = 41
THRESHOLD_VALUE = 15
MIN_AREA = 120

# --------------------------------------------------
# Create Results Folders
# --------------------------------------------------
base_results_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/results"
train_results_path = os.path.join(base_results_path, "training")
val_results_path = os.path.join(base_results_path, "validation")

os.makedirs(train_results_path, exist_ok=True)
os.makedirs(val_results_path, exist_ok=True)

# --------------------------------------------------
# Segmentation Function
# --------------------------------------------------
def segment_vessels(image):

    green = image[:, :, 1]

    blur_large = cv2.GaussianBlur(green, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)

    corrected = cv2.subtract(blur_large, green)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    # Circular mask
    h, w = green.shape
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, (w//2, h//2), min(h,w)//2 - 10, 255, -1)
    corrected = cv2.bitwise_and(corrected, corrected, mask=mask_circle)

    _, binary = cv2.threshold(corrected, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)

    cleaned = np.zeros_like(closing)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            cleaned[labels == i] = 255

    return cleaned


# --------------------------------------------------
# Evaluation + Saving Function
# --------------------------------------------------
def evaluate_and_save(images_path, masks_path, save_path):

    image_files = sorted(os.listdir(images_path))

    dice_scores = []
    jaccard_scores = []

    for file in image_files:

        img_path = os.path.join(images_path, file)
        mask_path = os.path.join(masks_path, file)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        segmented = segment_vessels(img)

        # ---- SAVE ONLY FINAL SEGMENTED IMAGE ----
        save_file_path = os.path.join(save_path, file)
        cv2.imwrite(save_file_path, segmented)

        # ---- Compute Metrics ----
        seg = segmented > 0
        gt  = gt_mask > 0

        intersection = np.logical_and(seg, gt)
        union = np.logical_or(seg, gt)

        dice = 2 * intersection.sum() / (seg.sum() + gt.sum() + 1e-8)
        jaccard = intersection.sum() / (union.sum() + 1e-8)

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)

    return dice_scores, jaccard_scores


# --------------------------------------------------
# TRAINING SET (200 Images)
# --------------------------------------------------
training_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/images"
training_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/masks"

train_dice, train_jaccard = evaluate_and_save(training_images_path,
                                              training_masks_path,
                                              train_results_path)

print("----- TRAINING SET RESULTS -----")
print("Images evaluated:", len(train_dice))
print("Average Dice:", np.mean(train_dice))
print("Average Jaccard:", np.mean(train_jaccard))
print()


# --------------------------------------------------
# VALIDATION SET (50 Images)
# --------------------------------------------------
validation_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/images"
validation_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/masks"

val_dice, val_jaccard = evaluate_and_save(validation_images_path,
                                          validation_masks_path,
                                          val_results_path)

print("----- VALIDATION SET RESULTS -----")
print("Images evaluated:", len(val_dice))
print("Average Dice:", np.mean(val_dice))
print("Average Jaccard:", np.mean(val_jaccard))