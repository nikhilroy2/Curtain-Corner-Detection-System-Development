import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # Apply adaptive thresholding with optimized parameters
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 5)
    return thresh

def detect_edges(preprocessed_image):
    # Apply Canny edge detection with optimized parameters
    edges = cv2.Canny(preprocessed_image, 30, 200)
    # Dilate edges to connect broken lines
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    # Apply morphological closing to fill small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed

def find_curtain_corners(image_path):
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
    
    # Find the largest contour (assuming it's the curtain)
    if not contours:
        raise ValueError("No contours found in the image")
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to get corners with stricter epsilon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Get the four corners
    corners = []
    if len(approx) >= 4:
        # Use convex hull to ensure we get the outermost points
        hull = cv2.convexHull(approx)
        # Convert to numpy array and reshape
        points = approx.reshape(-1, 2)
        
        # Get the four extreme corners
        # Top-left: point with smallest sum of coordinates
        top_left = points[np.argmin(points.sum(axis=1))]
        # Bottom-right: point with largest sum of coordinates
        bottom_right = points[np.argmax(points.sum(axis=1))]
        # Top-right: point with largest difference of coordinates
        top_right = points[np.argmax(points[:,0] - points[:,1])]
        # Bottom-left: point with smallest difference of coordinates
        bottom_left = points[np.argmin(points[:,0] - points[:,1])]
        
        corners = [top_left, top_right, bottom_right, bottom_left]
    
    # Visualize results
    result = image.copy()
    if corners:
        # Draw corners
        for corner in corners:
            cv2.circle(result, tuple(corner), 5, (0, 255, 0), -1)
        # Draw contour
        cv2.drawContours(result, [largest_contour], -1, (0, 0, 255), 2)
    
    return corners, result

def main():
    # Test images
    test_images = ['1_Color.png', '2_Color.png', '3_Color.png']
    
    for image_path in test_images:
        try:
            # Detect corners
            corners, result = find_curtain_corners(image_path)
            
            # Display results
            plt.figure(figsize=(12, 6))
            
            # Original image
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Result with detected corners
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('Detected Corners')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print corner coordinates
            print(f"\nCorner coordinates for {image_path}:")
            corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
            for name, corner in zip(corner_names, corners):
                print(f"{name}: {corner}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()