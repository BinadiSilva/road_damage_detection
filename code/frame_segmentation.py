import cv2
import numpy as np
import os
import glob

def get_refined_segmentation_mask(enhanced_frame):
    """
    Refined segmentation pipeline that ignores road texture 
    and focuses only on major structural damage.
    """
    # 1. Grayscale Conversion
    if len(enhanced_frame.shape) == 3:
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced_frame.copy()

    # 2. CRITICAL FIX: Texture Removal (Heavy Median Blur)
    # A large kernel (9x9) removes the grainy asphalt texture but keeps thick cracks and potholes intact.
    smoothed = cv2.medianBlur(gray, 9)

    # 3. STRICTER Adaptive Thresholding
    # - Block size increased to 51 (looks at a wider area to avoid small local shadows)
    # - C value increased to 10 (forces the algorithm to only pick pixels that are significantly darker than the road)
    binary_mask = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )

    # 4. Morphological Operations
    # Using a slightly larger kernel (5x5) to aggressively clean up remaining noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2 iterations of closing to ensure broken crack segments are stitched back together
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Connected Component Analysis & Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    final_segmented_mask = np.zeros_like(cleaned_mask)

    height, width = gray.shape
    max_allowed_area = (height * width) * 0.30  # Damage shouldn't exceed 30% of the image frame

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Minimum area increased to 150 to ignore any surviving gravel noise
        if 150 <= area <= max_allowed_area:
            final_segmented_mask[labels == i] = 255

    return final_segmented_mask

def main():
    # --- CONFIGURE YOUR FOLDERS HERE ---
    INPUT_FOLDER = "/Users/binadisilva/Desktop/RUSL/IP/Project_IP/Enhanced/patch_failure"    
    OUTPUT_FOLDER = "/Users/binadisilva/Desktop/RUSL/IP/Project_IP/patch_failure"  

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))

    if len(image_paths) == 0:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(image_paths)} images. Drawing refined contours...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Read the enhanced image
        enhanced_img = cv2.imread(img_path)
        if enhanced_img is None:
            continue
            
        # 1. Get the refined, texture-free mask
        mask = get_refined_segmentation_mask(enhanced_img)
        
        # 2. Convert to BGR for drawing colored lines
        if len(enhanced_img.shape) == 2:
            display_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = enhanced_img.copy()

        # 3. Find the contours of the true damage
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Draw the contours (Red color, thickness 2)
        cv2.drawContours(display_img, contours, -1, (0, 0, 255), 2)
        
        # Save the properly marked image
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), display_img)
        print(f"Refined contours marked and saved: {filename}")

    print(f"Done! Check '{OUTPUT_FOLDER}' for the much cleaner results.")

if __name__ == "__main__":
    main()