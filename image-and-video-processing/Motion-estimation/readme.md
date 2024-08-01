# Image and Video Processing

## Overview

This Task involves implementing several advanced image and video processing techniques including Principal Component Analysis (PCA), Discrete Cosine Transform (DCT), and motion estimation. The tasks cover image compression, decorrelation, and motion vector estimation.

## Tasks

### 1. Optimal Decorrelating Transform

- **Objective:** Implement Principal Component Analysis (PCA) to decorrelate an image.
- **Steps:**
  1. Divide the image into non-overlapping 8x8 patches and vectorize these patches into 64-dimensional vectors.
  2. Compute the covariance matrix of these vectors, ensuring the data is zero-mean.
  3. Reconstruct the image at three compression levels (48, 32, and 16 dimensions) and display the results.
  4. Assess the effectiveness of PCA in decorrelating images.

### 2. Discrete Cosine Transform (DCT)

- **Objective:** Implement a JPEG-like image encoder and decoder.
- **Encoder Steps:**
  1. Divide the image into 8x8 pixel blocks.
  2. Apply Type-II DCT to each block.
  3. Quantize each block using a quantization matrix.
- **Decoder Steps:**
  1. Dequantize each block by multiplying with the inverse quantization matrix.
  2. Apply Type-II IDCT to each block.
  3. Combine the blocks to reconstruct the image.
  4. Display the original and compressed images to evaluate the effectiveness of DCT for image compression.

### 3. Motion Estimation

- **Objective:** Implement motion estimation using the 3-step search algorithm.
- **Steps:**
  1. Find motion vectors using Mean Absolute Distance (MAD) and the 3-step search:
     - Step 1: Search ±4 pixels around the current macroblock.
     - Step 2: Refine search ±2 pixels around the best match from Step 1.
     - Step 3: Further refine search ±1 pixel around the best match from Step 2.
  2. Plot the motion vectors at each macroblock.
  3. Generate and display the motion-compensated predicted frame using the motion vectors and the first frame.
  4. Compute and display the error between the second frame and its motion-compensated predicted version.
