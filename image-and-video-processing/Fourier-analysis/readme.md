# EE6310 Image and Video Processing

## Overview

This Task involves implementing and testing various image processing techniques:

1. **Point Operations:** Linear transformations \( J = P*I \), \( J = I + L \), \( J = P*I + L \).
2. **Full Scale Contrast Stretch (FSCS):** Apply FSCS and compare histograms before and after.
3. **Log Magnitude Compression:** Apply log magnitude compression followed by FSCS and compare histograms.
4. **Gamma Correction:** Apply gamma correction with \( gamma = 1.4 \) and clip values, then compare histograms.
5. **Histogram Flattening:** Flatten histograms and compare before and after.

### Image Zooming

- Zoom images by 1.5x using Nearest Neighbor and Bilinear Interpolation.
- Compare quality and computational complexity.

### 2D Discrete Fourier Transform (DFT)

- Implement DFT and IDFT.
- Analyze and display image, its DFT magnitude, phase, and log-transformed magnitude.
- Demonstrate the effect of phase changes on image reconstruction.
