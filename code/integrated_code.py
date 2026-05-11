import os
import cv2
import numpy as np

VIDEO_PATH = "/Users/binadisilva/road_damage_detection/Videos/Video.mp4"

BASE_OUTPUT_FOLDER = "/Users/binadisilva/road_damage_detection"

FRAME_INTERVAL = 10   # save every 10th frame

# OUTPUT FOLDERS
ORIGINAL_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "Frames")
ENHANCED_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "Enhanced")
MASK_FOLDER     = os.path.join(BASE_OUTPUT_FOLDER, "Segmented")
MARKED_FOLDER   = os.path.join(BASE_OUTPUT_FOLDER, "Final_Output")

os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(MARKED_FOLDER, exist_ok=True)


# ENHANCEMENT FUNCTIONS
def detect_noise_type(gray, sp_ratio_thr=0.01, std_thr=25):
    total = gray.size
    sp_ratio = (np.sum(gray == 0) + np.sum(gray == 255)) / total

    if sp_ratio > sp_ratio_thr:
        return "salt_pepper"

    if np.std(gray) > std_thr:
        return "gaussian"

    return None


def is_low_contrast(gray, range_thr=60):
    return (int(gray.max()) - int(gray.min())) < range_thr


def shadow_present(gray, thr=25):
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    return (int(bg.max()) - int(bg.min())) > thr


def is_blurry(gray, lap_thr=100):
    return cv2.Laplacian(gray, cv2.CV_64F).var() < lap_thr


def balance_light_shadow(gray, strength=0.90):
    bg = cv2.GaussianBlur(gray, (71, 71), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)

    bg = np.clip(bg, 10.0, 255.0)
    corrected = (gray_f / bg) * np.mean(bg)
    corrected = np.clip(corrected, 0, 240).astype(np.uint8)

    out = cv2.addWeighted(gray, 1.0 - strength, corrected, strength, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_frame(frame):
    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise check and denoise
    noise_type = detect_noise_type(gray)

    if noise_type == "salt_pepper":
        denoised = cv2.medianBlur(gray, 5)
    elif noise_type == "gaussian":
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        denoised = gray.copy()

    # Step 3: Contrast enhancement
    if is_low_contrast(denoised):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
    else:
        contrast = denoised.copy()

    # Step 4: Shadow / light balance
    if shadow_present(contrast):
        balanced = balance_light_shadow(contrast, strength=0.90)
    else:
        balanced = contrast.copy()

    # Step 5: Sharpening
    if is_blurry(balanced):
        blur_small = cv2.GaussianBlur(balanced, (3, 3), 0)
        sharpened = cv2.addWeighted(balanced, 2.0, blur_small, -1.0, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        sharpened = balanced.copy()

    return sharpened


# SEGMENTATION FUNCTIONS
def get_segmentation_mask(enhanced_frame):
    
    # Step 1: Ensure grayscale
    if len(enhanced_frame.shape) == 3:
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced_frame.copy()

    # Step 2: Median blur to reduce road texture
    smoothed = cv2.medianBlur(gray, 9)

    # Step 3: Adaptive thresholding
    binary_mask = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10
    )

    # Step 4: Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Step 5: Connected component filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned_mask,
        connectivity=8
    )

    final_mask = np.zeros_like(cleaned_mask)

    height, width = gray.shape
    max_allowed_area = (height * width) * 0.30

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if 150 <= area <= max_allowed_area:
            final_mask[labels == i] = 255

    return final_mask


def draw_damage_contours(enhanced_frame, mask):
    # Convert enhanced grayscale image to BGR for colored contour drawing
    if len(enhanced_frame.shape) == 2:
        marked_img = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
    else:
        marked_img = enhanced_frame.copy()

    # Find contours from the segmented mask
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw damage contours in red
    cv2.drawContours(marked_img, contours, -1, (0, 0, 255), 2)

    return marked_img


# MAIN INTEGRATED PIPELINE

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError("Error opening video file. Check VIDEO_PATH")

frame_index = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extract every nth frame
    if frame_index % FRAME_INTERVAL == 0:
        filename = f"frame_{saved_count:05d}.jpg"

        # 1. Save original extracted frame
        original_path = os.path.join(ORIGINAL_FOLDER, filename)
        cv2.imwrite(original_path, frame)

        # 2. Enhance frame
        enhanced = enhance_frame(frame)
        enhanced_path = os.path.join(ENHANCED_FOLDER, filename)
        cv2.imwrite(enhanced_path, enhanced)

        # 3. Segment enhanced frame
        mask = get_segmentation_mask(enhanced)
        mask_path = os.path.join(MASK_FOLDER, filename)
        cv2.imwrite(mask_path, mask)

        # 4. Draw contours / final marked output
        marked = draw_damage_contours(enhanced, mask)
        marked_path = os.path.join(MARKED_FOLDER, filename)
        cv2.imwrite(marked_path, marked)

        saved_count += 1

    frame_index += 1

cap.release()

print("Final integrated processing completed successfully.")
print("Original frames saved in:", ORIGINAL_FOLDER)
print("Enhanced frames saved in:", ENHANCED_FOLDER)
print("Segmented masks saved in:", MASK_FOLDER)
print("Final marked outputs saved in:", MARKED_FOLDER)
print("Total processed frames:", saved_count)