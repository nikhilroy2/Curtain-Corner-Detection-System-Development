import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    # Apply adaptive thresholding with refined parameters
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    # Perform morphological operations with larger kernel
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def detect_edges(preprocessed_image):
    # Enhanced vertical line detection with adjusted parameters
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, vertical_kernel)
    vertical = cv2.dilate(vertical, np.ones((5,1), np.uint8), iterations=2)
    
    # Enhanced horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal = cv2.dilate(horizontal, np.ones((1,3), np.uint8), iterations=1)
    
    # Combine and enhance edges
    combined = cv2.bitwise_or(vertical, horizontal)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
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
    
    # Detect edges with enhanced parameters for lower edge detection
    edges = detect_edges(preprocessed)
    
    # Apply additional processing to enhance lower edges
    lower_kernel = np.ones((7,1), np.uint8)  # Increased kernel size
    lower_edges = cv2.dilate(edges[height//2:], lower_kernel, iterations=2)  # Increased iterations
    edges[height//2:] = lower_edges
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and shape with adjusted parameters
    min_area = height * width * 0.005  # Reduced minimum area threshold
    max_area = height * width * 0.8    # Increased maximum area threshold
    panel_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            
            # Calculate aspect ratio
            _, (w, h), _ = rect
            aspect_ratio = max(w/h if h != 0 else 0, h/w if w != 0 else 0)
            
            if 1.5 < aspect_ratio < 10:  # Relaxed aspect ratio constraints
                # Enhanced lower corner detection
                # Get the bottom points with sub-pixel accuracy
                bottom_points = box[box[:, 1].argsort()][-2:]
                refined_bottom = []
                
                for point in bottom_points:
                    # Create larger ROI around the point for better analysis
                    roi_size = 25  # Further increased ROI size for better precision
                    x, y = point.astype(int)
                    roi = edges[max(0, y-roi_size):min(height, y+roi_size),
                               max(0, x-roi_size):min(width, x+roi_size)]
                    
                    # Refine corner position using weighted centroid with enhanced precision
                    if roi.size > 0:
                        y_coords, x_coords = np.nonzero(roi)
                        if len(x_coords) > 0:
                            # Apply enhanced Gaussian weights with stronger distance penalty
                            center_y, center_x = roi_size, roi_size
                            dist_weights = np.exp(-0.8 * ((x_coords - center_x)**2 + (y_coords - center_y)**2) / (roi_size/2)**2)
                            edge_weights = roi[y_coords, x_coords].astype(float) / 255.0
                            weights = dist_weights * edge_weights * edge_weights  # Square edge weights for stronger edge influence
                            
                            refined_x = np.average(x_coords, weights=weights) + max(0, x-roi_size)
                            refined_y = np.average(y_coords, weights=weights) + max(0, y-roi_size)
                            refined_bottom.append([int(refined_x), int(refined_y)])
                        else:
                            refined_bottom.append([x, y])
                    else:
                        refined_bottom.append([x, y])
                
                # Update the bottom corners in the box with refined positions
                box[box[:, 1].argsort()][-2:] = refined_bottom
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
        
        # Draw corner points with emphasis on lower corners
        for i, corner in enumerate(corners):
            if i >= 2:  # Lower corners
                # Draw enhanced markers for lower corners
                cv2.circle(result, corner, 12, (0, 0, 255), -1)  # Larger red fill
                cv2.circle(result, corner, 12, (0, 0, 0), 2)  # Black outline
                cv2.drawMarker(result, corner, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)  # Yellow cross marker
            else:  # Upper corners
                cv2.circle(result, corner, 4, (0, 0, 255), -1)
                cv2.circle(result, corner, 4, (0, 0, 0), 2)
        
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

            # Get preprocessed image
            preprocessed = preprocess_image(original)
            
            # Detect corners
            corners, result = find_panel_corners(image_path)
            
            # Create a copy of preprocessed image for visualization
            preprocessed_viz = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            
            # Draw the same boxes and corners on preprocessed image
            for idx, box in enumerate(corners):
                # Draw panel outline
                box_array = np.array(box)
                cv2.drawContours(preprocessed_viz, [box_array], 0, (0, 255, 0), 2)
                
                # Draw corner points
                for i, corner in enumerate(box):
                    if i >= 2:  # Lower corners
                        cv2.circle(preprocessed_viz, corner, 12, (0, 0, 255), -1)
                        cv2.circle(preprocessed_viz, corner, 12, (0, 0, 0), 2)
                        cv2.drawMarker(preprocessed_viz, corner, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    else:  # Upper corners
                        cv2.circle(preprocessed_viz, corner, 4, (0, 0, 255), -1)
                        cv2.circle(preprocessed_viz, corner, 4, (0, 0, 0), 2)
                
                # Add panel number
                x, y = box[0]
                cv2.putText(preprocessed_viz, f'Panel {idx + 1}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Display results
            plt.figure(figsize=(12, 5))
            
            # Preprocessed image with boxes
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(preprocessed_viz, cv2.COLOR_BGR2RGB))
            plt.title('Preprocessed Image with Detection')
            plt.axis('off')
            
            # Final result with detected panels and corners
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('Detected Panels and Corners')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()