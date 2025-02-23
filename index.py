import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter with optimized parameters for better noise reduction
    blurred = cv2.bilateralFilter(gray, 11, 100, 100)
    # Apply adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Apply Canny edge detection with optimized thresholds
    edges = cv2.Canny(thresh, 50, 150)
    # Dilate to connect edge components
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    return dilated

def detect_edges(preprocessed_image):
    # Find vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, vertical_kernel)
    
    # Find horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Combine vertical and horizontal lines
    combined = cv2.bitwise_or(vertical, horizontal)
    
    # Clean up the combined edges
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def find_panel_corners(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Store original image dimensions
    height, width = image.shape[:2]
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Detect edges
    edges = detect_edges(preprocessed)
    
    # Find contours with adjusted parameters
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio with adjusted thresholds
    min_area = height * width * 0.02  # Minimum area as 2% of image
    max_area = height * width * 0.5   # Maximum area as 50% of image
    panel_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h/w
            if 1.2 < aspect_ratio < 12:  # Even more flexible aspect ratio for curtain panels
                # Use convex hull to get better shape
                hull = cv2.convexHull(contour)
                # Approximate the contour with more precise epsilon
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) >= 4:  # Only add if it has at least 4 corners
                    panel_contours.append(approx)
    
    # Sort panels from left to right
    panel_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    
    # Process each panel
    result = image.copy()
    all_corners = []
    
    for idx, contour in enumerate(panel_contours):
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort points to ensure consistent order
        # Sort by y first to separate top and bottom points
        box = box[box[:, 1].argsort()]
        top = box[:2]
        bottom = box[2:]
        
        # Sort top and bottom points by x
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]
        
        # Combine points in order: top-left, top-right, bottom-right, bottom-left
        box = np.vstack((top, bottom[::-1]))
        
        # Draw thick border around panel with gradient color
        cv2.drawContours(result, [box], 0, (0, 0, 255), 5)
        
        # Store corners
        corners = [(int(x), int(y)) for x, y in box]
        all_corners.append(corners)
        
        # Draw corner points with enhanced visibility
        for corner in corners:
            # Draw white background circle for better contrast
            cv2.circle(result, corner, 12, (255, 255, 255), -1)
            # Draw larger filled green circle
            cv2.circle(result, corner, 10, (0, 255, 0), -1)
            # Draw border for better visibility
            cv2.circle(result, corner, 10, (0, 100, 0), 2)
        
        # Add panel number
        x, y = corners[0]  # Use top-left corner for label
        cv2.putText(result, f'Panel {idx + 1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return all_corners, result

def main():
    # Test images
    test_images = ['1_Color.png', '2_Color.png', '3_Color.png']
    
    for image_path in test_images:
        try:
            # Read original image
            original = cv2.imread(image_path)
            if original is None:
                print(f"Could not read image: {image_path}")
                continue

            # Get preprocessed image and edges for debugging
            preprocessed = preprocess_image(original)
            edges = detect_edges(preprocessed)
            
            # Detect corners
            corners, result = find_panel_corners(image_path)
            
            # Display results with intermediate steps
            plt.figure(figsize=(15, 10))
            
            # Original image
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Preprocessed image
            plt.subplot(222)
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Preprocessed Image')
            plt.axis('off')
            
            # Edge detection
            plt.subplot(223)
            plt.imshow(edges, cmap='gray')
            plt.title('Edge Detection')
            plt.axis('off')
            
            # Final result
            plt.subplot(224)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('Detected Panels and Corners')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print corner coordinates and contour info
            print(f"\nResults for {image_path}:")
            if len(corners) == 0:
                print("No panels detected!")
            else:
                for i, panel_corners in enumerate(corners):
                    print(f"\nPanel {i + 1}:")
                    corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                    for name, corner in zip(corner_names, panel_corners):
                        print(f"{name}: {corner}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()