# Curtain Corner Detection System

An advanced computer vision system that automatically detects and marks the corners of curtains in images using OpenCV and Python.

## Features

- Automatic detection of curtain corners in images
- Robust preprocessing pipeline for image enhancement
- Advanced edge detection and contour analysis
- Visual output with marked corners and contours
- Corner coordinate extraction and reporting

## Requirements

The system requires Python 3.x and the following dependencies:

```
opencv-python==4.11.0.86
numpy==2.2.3
matplotlib==3.10.0
```

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd curtain-corner-detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your curtain images in the project directory
2. Run the detection script:
   ```bash
   python index.py
   ```

The script will process each test image and:
- Display the original image alongside the processed image with detected corners
- Print the coordinates of detected corners
- Mark corners with green dots and contours with red lines

## How It Works

The system uses a multi-stage approach to detect curtain corners:

1. **Preprocessing**:
   - Conversion to grayscale
   - Bilateral filtering for noise reduction
   - Contrast enhancement using CLAHE
   - Adaptive thresholding

2. **Edge Detection**:
   - Canny edge detection
   - Edge dilation and morphological operations

3. **Corner Detection**:
   - Contour detection and filtering
   - Convex hull computation
   - Extreme point analysis for corner identification

## Sample Output

For each processed image, the system outputs:
- Visual display showing original and processed images
- Corner coordinates in the format:
  ```
  Corner coordinates for image.png:
  Top-Left: [x, y]
  Top-Right: [x, y]
  Bottom-Right: [x, y]
  Bottom-Left: [x, y]
  ```

## Error Handling

The system includes robust error handling for common issues:
- Invalid image paths
- Unreadable images
- Images without detectable contours

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.