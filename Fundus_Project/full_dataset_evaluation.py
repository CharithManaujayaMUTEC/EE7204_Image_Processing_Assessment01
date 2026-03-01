import cv2
import numpy as np
import os

# --------------------------------------------------
# PARAMETERS TO SEARCH
# --------------------------------------------------
GAUSSIAN_KERNELS = [51, 61]
THRESHOLD_VALUES = [8, 10, 12]
DILATION_ITERATIONS = [1, 2]

# --------------------------------------------------
# Segmentation Function
# --------------------------------------------------
def segment_vessels(image, gaussian_kernel, threshold_value, dilation_iter):

    # 1️⃣ Extract green channel
    green = image[:, :, 1]

    # 2️⃣ Illumination correction (large Gaussian blur)
    blur_large = cv2.GaussianBlur(
        green,
        (gaussian_kernel, gaussian_kernel),
        0
    )

    corrected = cv2.subtract(blur_large, green)

    # 3️⃣ Normalize to full intensity range
    corrected = cv2.normalize(
        corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # 4️⃣ Apply circular mask (remove black borders)
    h, w = green.shape
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        mask_circle,
        (w // 2, h // 2),
        min(h, w) // 2 - 10,
        255,
        -1
    )

    corrected = cv2.bitwise_and(
        corrected,
        corrected,
        mask=mask_circle
    )

    # 5️⃣ Global threshold
    _, binary = cv2.threshold(
        corrected,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # 6️⃣ Dilation (recover thin vessels)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    dilated = cv2.dilate(
        binary,
        kernel,
        iterations=dilation_iter
    )

    # 7️⃣ Closing (connect broken vessels)
    closed = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed


# --------------------------------------------------
# Evaluation Function (Dice + Jaccard)
# --------------------------------------------------
def evaluate_dataset(images_path,
                     masks_path,
                     gaussian_kernel,
                     threshold_value,
                     dilation_iter):

    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    dice_scores = []
    jaccard_scores = []

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

        segmented = segment_vessels(
            img,
            gaussian_kernel,
            threshold_value,
            dilation_iter
        )

        seg = segmented > 0
        gt = gt_mask > 0

        intersection = np.logical_and(seg, gt)
        union = np.logical_or(seg, gt)

        dice = (
            2 * intersection.sum()
            / (seg.sum() + gt.sum() + 1e-8)
        )

        jaccard = (
            intersection.sum()
            / (union.sum() + 1e-8)
        )

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)

    return np.mean(dice_scores), np.mean(jaccard_scores)


# --------------------------------------------------
# GRID SEARCH ON TRAINING SET
# --------------------------------------------------
training_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/images"
training_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/training_set/masks"

best_dice = 0
best_jaccard = 0
best_params = None

print("Running Final Optimized Search...\n")

for gk in GAUSSIAN_KERNELS:
    for T in THRESHOLD_VALUES:
        for d in DILATION_ITERATIONS:

            avg_dice, avg_jaccard = evaluate_dataset(
                training_images_path,
                training_masks_path,
                gk,
                T,
                d
            )

            print(f"Kernel={gk}, T={T}, Dil={d} → "
                  f"Dice={avg_dice:.4f}, "
                  f"Jaccard={avg_jaccard:.4f}")

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_jaccard = avg_jaccard
                best_params = (gk, T, d)


print("\nBest Training Parameters:")
print("Gaussian Kernel:", best_params[0])
print("Threshold:", best_params[1])
print("Dilation Iter:", best_params[2])
print("Best Training Dice:", best_dice)
print("Best Training Jaccard:", best_jaccard)


# --------------------------------------------------
# VALIDATION SET EVALUATION
# --------------------------------------------------
validation_images_path = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/images"
validation_masks_path  = "/content/EE7204_Image_Processing_Assessment01/Fundus_Project/validation_set/masks"

val_dice, val_jaccard = evaluate_dataset(
    validation_images_path,
    validation_masks_path,
    best_params[0],
    best_params[1],
    best_params[2]
)

print("\nValidation Dice:", val_dice)
print("Validation Jaccard:", val_jaccard)