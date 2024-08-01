# Image Processing Techniques - Convolution and Filtering

## Overview

This Task involves implementing and experimenting with various convolution and filtering techniques in image processing.

## Tasks

### 1. Periodic Convolution

1. **Circular Convolution:** Implement circular convolution of two images. Test using a 5x5 averaging filter and verify the result using IDFT of the product of the image and filter DFT.
2. **Linear Convolution:** Perform linear convolution by zero padding the image and filter and using the circular convolution function.
3. **Filter Size Comparison:** Experiment with different filter sizes and compare the efficiency of direct linear convolution versus IDFT-DFT approach with zero padding.

### 2. Linear Image Filters

1. **Averaging Filter:** Experiment with different parameters of the averaging filter. Plot the DFT magnitude spectrum and analyze its low-pass filter characteristics.
2. **Gaussian Filter:** Test different parameters of the Gaussian filter. Plot the DFT magnitude spectrum and analyze its low-pass filter characteristics.
3. **Band Pass Filter:** Create a band pass filter by subtracting two Gaussian filters. Verify its characteristics using the DFT magnitude spectrum and experiment with different parameter values.
4. **Laplacian Filter:** Plot the DFT magnitude spectrum of a Laplacian filter and verify its high-pass filter characteristics.

### 3. Applications of Linear Filters

1. **Denoising:** Apply averaging and Gaussian filters to denoise images with AWGN (σ² = 25). Vary filter sizes and standard deviations to observe the trade-offs in denoising.
2. **Edge Detection and Enhancement:** Apply band pass and high pass filters to the mandrill image from the USC dataset. Observe and tabulate the results for different filter parameters.

