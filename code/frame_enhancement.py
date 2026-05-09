import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

INPUT_FOLDER  = "InputFolder"
OUTPUT_FOLDER = "Output Folder"
#INPUT_FOLDER  = "/Users/binadisilva/Downloads"
#OUTPUT_FOLDER = "/Users/binadisilva/Desktop/RUSL/IP/Project_IP/Enhanced/crack"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- FILES (.jpg) ----------------
image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
#image_files = ["/Users/binadisilva/Downloads/WhatsApp Image 2026-03-02 at 13.57.27.jpeg"]
if len(image_files) == 0:
    raise ValueError("No .jpg images found in input folder")

# ---------------- HISTOGRAM ----------------
def calc_hist(gray):
    return cv2.calcHist([gray], [0], None, [256], [0, 256])

# ---------------- CHECK FUNCTIONS ----------------
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

# ---------------- SHADOW BALANCE ----------------
def balance_light_shadow(gray, strength=0.90):
    bg = cv2.GaussianBlur(gray, (71, 71), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)

    bg = np.clip(bg, 10.0, 255.0)
    corrected = (gray_f / bg) * np.mean(bg)
    corrected = np.clip(corrected, 0, 240).astype(np.uint8)

    out = cv2.addWeighted(gray, 1.0 - strength, corrected, strength, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------- ENHANCE ONE IMAGE (GRAYSCALE OUTPUT) ----------------
def enhance_one(bgr):
    # original grayscale (this is the "Gray" stage)
    gray_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # -------- Noise Check --------
    ntype = detect_noise_type(gray_orig)

    # -------- Denoise --------
    denoise_used = False
    denoise_name = "Denoise: Skipped"
    if ntype == "salt_pepper":
        gray1 = cv2.medianBlur(gray_orig, 5)
        denoise_name = "Denoise: Median Blur (Salt & Pepper)"
        denoise_used = True
    elif ntype == "gaussian":
        gray1 = cv2.GaussianBlur(gray_orig, (5, 5), 0)
        denoise_name = "Denoise: Gaussian Blur (Gaussian Noise)"
        denoise_used = True
    else:
        gray1 = gray_orig.copy()

    # -------- Contrast --------
    contrast_used = False
    contrast_name = "Contrast: Skipped"
    if is_low_contrast(gray1):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray2 = clahe.apply(gray1)
        contrast_name = "Contrast: CLAHE"
        contrast_used = True
    else:
        gray2 = gray1.copy()

    # -------- Shadow Removal / Light Balance --------
    shadow_used = False
    shadow_name = "Shadow Removal: Skipped"
    if shadow_present(gray2):
        gray3 = balance_light_shadow(gray2, strength=0.90)
        shadow_name = "Shadow Removal: Background Normalization"
        shadow_used = True
    else:
        gray3 = gray2.copy()

    # -------- Sharpen --------
    sharp_used = False
    sharp_name = "Sharpen: Skipped"
    if is_blurry(gray3):
        blur_small = cv2.GaussianBlur(gray3, (3, 3), 0)
        gray4 = cv2.addWeighted(gray3, 2.0, blur_small, -1.0, 0)
        gray4 = np.clip(gray4, 0, 255).astype(np.uint8)
        sharp_name = "Sharpen: Unsharp Mask"
        sharp_used = True
    else:
        gray4 = gray3.copy()

    gray_final = gray4.copy()

    # -------- Demo stages (ONLY include steps that were actually DONE) --------
    stages = []
    stages.append(("Original (Gray)", gray_orig, calc_hist(gray_orig)))

    # Always show "Gray" stage as same as original grayscale (kept for pipeline completeness)
    stages.append(("Gray (Grayscale Conversion)", gray_orig, calc_hist(gray_orig)))

    if denoise_used:
        stages.append((denoise_name, gray1, calc_hist(gray1)))
    if contrast_used:
        stages.append((contrast_name, gray2, calc_hist(gray2)))
    if shadow_used:
        stages.append((shadow_name, gray3, calc_hist(gray3)))
    if sharp_used:
        stages.append((sharp_name, gray4, calc_hist(gray4)))

    stages.append(("Final Enhanced (Gray)", gray_final, calc_hist(gray_final)))

    # names of applied techniques (for printing)
    used_steps = [
        denoise_name,
        contrast_name,
        shadow_name,
        sharp_name
    ]

    # return also the main step outputs for the final "sample step images" display
    step_images = {
        "original": gray_orig,
        "gray": gray_orig,
        "denoise": gray1,
        "contrast": gray2,
        "shadow": gray3,
        "sharpen": gray4,
        "final": gray_final
    }

    step_used = {
        "denoise": denoise_used,
        "contrast": contrast_used,
        "shadow": shadow_used,
        "sharpen": sharp_used
    }

    return gray_final, stages, gray_orig, gray_final, used_steps, step_images, step_used

# --------------- DEMO ON ONE SAMPLE IMAGE ------------------
sample_path = image_files[0]
sample_bgr = cv2.imread(sample_path)
if sample_bgr is None:
    raise ValueError("Sample image could not be loaded")

final_gray, stages, orig_gray, final_gray_img, used_steps, step_images, step_used = enhance_one(sample_bgr)

for s in used_steps:
    print(s)

# ---------------- SHOW STEP IMAGES + HISTOGRAMS (ONLY DONE STEPS) ----------------
rows = len(stages)
plt.figure(figsize=(12, 3 * rows))

for i, (title, im, hist) in enumerate(stages, start=1):
    # image
    plt.subplot(rows, 2, (i - 1) * 2 + 1)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")

    # histogram
    plt.subplot(rows, 2, (i - 1) * 2 + 2)
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlim([0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# END OUTPUT: SAMPLE 1 IMAGE OF EACH STEP (ONLY IF DONE)
# (Original, Gray always shown; others only if applied)
sample_titles = []
sample_imgs = []

# original + gray always included
sample_titles.append("Original (Gray)")
sample_imgs.append(step_images["original"])

sample_titles.append("Gray")
sample_imgs.append(step_images["gray"])

# include only if used
if step_used["denoise"]:
    sample_titles.append("Denoise")
    sample_imgs.append(step_images["denoise"])

if step_used["contrast"]:
    sample_titles.append("Contrast")
    sample_imgs.append(step_images["contrast"])

if step_used["shadow"]:
    sample_titles.append("Light Balance")
    sample_imgs.append(step_images["shadow"])

if step_used["sharpen"]:
    sample_titles.append("Sharpen")
    sample_imgs.append(step_images["sharpen"])

# final always included
sample_titles.append("Final Enhanced")
sample_imgs.append(step_images["final"])

cols = len(sample_imgs)
plt.figure(figsize=(4 * cols, 4))

for i in range(cols):
    plt.subplot(1, cols, i + 1)
    plt.imshow(sample_imgs[i], cmap="gray")
    plt.title(sample_titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# -------------- FINAL REQUIRED: ORIGINAL VS ENHANCED IMAGE --------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(orig_gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(final_gray_img, cmap="gray")
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------- FINAL REQUIRED: ORIGINAL HISTOGRAM VS ENHANCED HISTOGRAM -------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(calc_hist(orig_gray))
plt.title("Original Histogram")
plt.xlim([0, 256])
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(calc_hist(final_gray_img))
plt.title("Enhanced Histogram")
plt.xlim([0, 256])
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# -------- APPLY TO ALL FRAMES AND SAVE (ONLY FINAL ENHANCED GRAYSCALE) --------
for p in image_files:
    img = cv2.imread(p)
    if img is None:
        continue

    out_gray, _, _, _, _, _, _ = enhance_one(img)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, os.path.basename(p)), out_gray)

print("Done. Grayscale enhanced images saved.")