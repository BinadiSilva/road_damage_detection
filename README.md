Video-Based Road Damage Detection Using Image Enhancement and Segmentation

ICT2403 – Graphics and Image Processing

Department of Information Technology
Faculty of Applied Sciences
Rajarata University of Sri Lanka

Group 03
* ICT/2023/012_6006_Perera_JAAD
* ICT/2023/014_6007_Bandara_MRV
* ICT/2023/015_6008_SILVA_GBD
* ICT/2023/078_6061_Gunarathna WAAI

1. Project Overview

This project presents a video-based road damage detection system developed using classical image processing techniques. The system processes road surface videos captured using mobile devices and performs:

* Frame extraction
* Image enhancement
* Road damage segmentation
* Final damage visualization

The main objective is to identify and isolate road surface degradations such as:

* Cracks
* Potholes
* Patch Failures
* Edge Cracking
* Road Marking Removal

2. Dataset and Video Acquisition

Videos were recorded under real-world environmental conditions from:

* Mihintale
* Kanadarawa
* Anuradhapura New Town
* Residential road areas

Recording Details

* Device: Apple iPhone 11
* Resolution: 1920 × 1080
* Frame Rate: 30 FPS
* Video Duration: 15–30 seconds

Damage Types Captured

* Crack
* Pothole
* Patch Failure
* Edge Cracking
* Road Marking Removal

3. Frame Extraction

Frames were extracted from videos using OpenCV.

Frame Selection Strategy

* Every 10th frame was extracted
* Frames saved in JPEG (.jpg) format
* Original resolution preserved

Extraction Pipeline

Video → Frame Sampling → JPEG Frames

4. Image Enhancement Pipeline

The enhancement stage improves frame quality before segmentation.

Enhancement Pipeline

Read → Gray → Noise Check → Denoise → Contrast → Shadow/Light Balance → Sharpen → Save

Techniques Used

* Grayscale Conversion
* Median Filtering
* Gaussian Filtering
* CLAHE Contrast Enhancement
* Illumination Normalization
* Unsharp Masking

Libraries Used

* OpenCV
* NumPy
* Matplotlib

5. Segmentation Pipeline

The segmentation stage isolates road damage regions from enhanced frames.

Segmentation Pipeline

Enhanced Frame → Thresholding → Morphological Operations → Connected Component Analysis → Final Segmented Output

Techniques Used

* Otsu Thresholding
* Adaptive Thresholding
* Morphological Opening
* Morphological Closing
* Connected Components
* Contour Detection

6. Experimental Evaluation

The system was tested using frames from multiple road degradation categories.

Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Intersection over Union (IoU)
* Dice Coefficient

Observations

* Potholes achieved highest segmentation accuracy
* Thin cracks were more challenging
* Shadows and uneven illumination affected performance
* Enhancement significantly improved segmentation quality

7. Project Structure

road_damage_detection/
│
├── Videos/
├── Frames/
├── Enhanced/
├── Segmented/
├── Final_Output/
│
├── code/
│   ├── frame_extraction.py
│   ├── enhancement_pipeline.py
│   ├── segmentation_pipeline.py
│   └── final_integration.py
│
├── reports/
├── presentation/
└── README.md

8. Required Libraries

Install required Python libraries:

pip install opencv-python numpy matplotlib

9. Running the Project

The files below are included separately for academic understanding and testing of individual stages:

* frame_extraction.py
* frame_enhancement.py
* frame_segmentation.py

These files demonstrate the implementation of:

* frame extraction
* image enhancement
* image segmentation

However, the main runnable program of the project is:

integrated_code.py

This integrated program combines:

* frame extraction
* enhancement
* segmentation
* final output generation

into a single processing pipeline.

Final Integrated Pipeline

Run the final system using:

python integrated_code.py

10. Final Outputs

The integrated system automatically processes the input video and generates:

* Extracted original frames
* Enhanced frames
* Segmented masks
* Final marked outputs

All outputs are saved into their corresponding folders automatically.

11. Limitations

The current implementation still faces challenges under:

* Strong shadows
* Motion blur
* Wet road reflections
* Complex road textures
* Low contrast conditions

12. Future Improvements

Possible future improvements include:

* Adaptive parameter tuning
* Deep learning-based segmentation
* Temporal video analysis
* Advanced shadow removal
* Hybrid segmentation methods

13. Academic Note

This project was developed for academic purposes under:

ICT2403 – Graphics and Image Processing
Department of Information Technology
Rajarata University of Sri Lanka

14. Repository

GitHub Repository:

https://github.com/BinadiSilva/road_damage_detection