# Image Quality Assessment and Edge Detection

## Overview

This Task focuses on image quality assessment, edge detection, and template matching. It involves both subjective and objective evaluations of image quality, implementation of edge detection techniques, and template matching.

## Tasks

### 1. Image Quality Assessment

1. **MSE and PSNR Calculation:**
   - Implement a function to compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between two images.

2. **SSIM Index Calculation:**
   - Implement a function to compute the Structural Similarity Index (SSIM) and its map using a Gaussian window.

3. **Distortions and Evaluation:**
   - Apply the following distortions to three images: mean shift, AWGN, JPEG compression, and Gaussian blur.
   - Compute MSE, PSNR, and SSIM for each distortion.
   - Subjectively rate the images and compute the correlation between subjective ratings and PSNR/SSIM.

### 2. Edge Detection

1. **Gradient-Based Edge Detection:**
   - Implement gradient operators: Centered 2-D differencing, Roberts, Prewitt, and Sobel.
   - Estimate gradient magnitude and threshold to find the edge map.

2. **Laplacian-Based Edge Detection:**
   - Compute the Laplacian using the given template and detect edges using zero crossing.

3. **Laplacian of Gaussian (LoG):**
   - Implement the LoG edge detector and compare results with gradient-based methods.

### 3. Template Matching

1. **Template Matching Implementation:**
   - Implement template matching using normalized cross-correlation.
   - Determine a suitable threshold for detecting matches.

2. **Noise and Rotation Effects:**
   - Add Gaussian noise to the template and analyze the effect on threshold.
   - Rotate the image and template, then analyze the effect of rotation on threshold.