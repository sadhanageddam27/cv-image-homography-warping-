# Image Homography and Warping

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange)
![scikit-image](https://img.shields.io/badge/scikit--image-0.21-blueviolet)

Computer Vision project implementing homography estimation from scratch,
perspective rectification, augmented reality texture insertion, document
scanning, and a side-by-side comparison of Triangular Mesh vs Thin-Plate
Spline (TPS) warping.

## Live Project Report
Full results with images and analysis:
**https://sadhanageddam27.github.io/project2/**

---

## Project Structure

```
cv-image-homography-warping/
├── code/
│   ├── part1.py          # Homography estimation + rectification
│   ├── part2.py          # Document scanning + AR texture insertion
│   └── part3.py          # Triangular Mesh vs TPS warping comparison
├── output_images/        # Result images from all three parts
└── report/
    └── Project_2_Report.pdf
```

---

## What This Project Covers

### Part 1 — Homography Estimation and Rectification
- Interactive 4-point selection (TL → TR → BR → BL) with undo support
- Homography matrix computed using **Normalized Direct Linear Transformation (DLT)** with SVD for numerical stability
- Inverse mapping with **bilinear interpolation** to produce fronto-parallel rectified output
- Tested on 5 different planar surfaces with consistent results

**Core math:**
```
H estimated via: A·h = 0 → SVD → last row of Vt reshaped to 3×3
Normalized to improve conditioning: T2_inv @ H_normalized @ T1
```

### Part 2 — Creative Applications
**Document Scanning**
- Rectify a skewed document photo into a top-down scanned copy
- Adaptive thresholding applied post-warp for a binary print-like output

**AR Texture Insertion**
- Warp and composite a logo/texture onto any planar surface in a scene
- Preserves correct perspective alignment using the same DLT pipeline

### Part 3 — Warping Comparison: Triangular Mesh vs TPS
- Tested on digit pairs: `3-a → 3-b` and `7-a → 7-b`
- 12–16 manual correspondences + 8 auto boundary points per pair
- Compared `PiecewiseAffineTransform` (scikit-image) vs `ThinPlateSplineTransform`

| Aspect | Triangular Mesh | Thin-Plate Spline |
|--------|----------------|-------------------|
| Type | Local per-triangle affine | Global smooth, min. bending energy |
| Continuity | Possible seams at triangle edges | Globally continuous |
| Smoothness | Piecewise-linear | Smooth curved transitions |
| Speed | Faster | Slightly slower (global solve) |
| Best for | Straight-edged shapes (digit 7) | Curved shapes (digit 3) |

---

## Key Results

- Perspective distortion fully corrected across all 5 test images
- AR texture composite aligns with surface perspective accurately
- TPS outperforms Mesh on curved shapes; Mesh is faster on linear geometry
- Custom NMS indices matched `torchvision.ops` exactly (verified)

---

## Setup and Usage

```bash
# Clone the repo
git clone https://github.com/sadhanageddam27/cv-image-homography-warping.git
cd cv-image-homography-warping

# Install dependencies
pip install opencv-python numpy matplotlib pillow scikit-image

# Run Part 1 — rectification
python code/part1.py

# Run Part 2 — document scan + AR
python code/part2.py

# Run Part 3 — warping comparison
python code/part3.py
```

Place your input images in a `part1_images/` folder before running Part 1.

---

## Tech Stack
Python · OpenCV · NumPy · scikit-image · Matplotlib · PIL

## Topics
`computer-vision` `homography` `image-warping` `perspective-rectification`
`augmented-reality` `document-scanning` `thin-plate-spline` `opencv` `python`
