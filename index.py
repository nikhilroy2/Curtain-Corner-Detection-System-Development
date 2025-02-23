import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Perform morphological operations
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def detect_edges(preprocessed_image):
    # Find vertical lines using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, vertical_kernel)
    
    # Find horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Combine vertical and horizontal lines
    combined = cv2.bitwise_or(vertical, horizontal)
    return combined

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
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and shape
    min_area = height * width * 0.01
    max_area = height * width * 0.5
    panel_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate aspect ratio
            _, (w, h), _ = rect
            aspect_ratio = max(w/h if h != 0 else 0, h/w if w != 0 else 0)
            
            if 2 < aspect_ratio < 8:  # Filter based on aspect ratio
                panel_contours.append(box)
    
    # Sort panels from left to right
    panel_contours.sort(key=lambda c: np.min(c[:, 0]))
    
    # Process each panel
    result = image.copy()
    all_corners = []
    
    for idx, box in enumerate(panel_contours):
        # Sort points to get consistent order (top-left, top-right, bottom-right, bottom-left)
        box = box[np.lexsort((box[:, 0], box[:, 1]))]
        if box[0][0] > box[1][0]:
            box[[0, 1]] = box[[1, 0]]
        if box[2][0] < box[3][0]:
            box[[2, 3]] = box[[3, 2]]
        
        # Draw panel outline
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
        
        # Store corners
        corners = [(int(x), int(y)) for x, y in box]
        all_corners.append(corners)
        
        # Draw corner points
        for corner in corners:
            cv2.circle(result, corner, 6, (0, 0, 255), -1)
            cv2.circle(result, corner, 6, (0, 0, 0), 2)
        
        # Add panel number
        x, y = corners[0]
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