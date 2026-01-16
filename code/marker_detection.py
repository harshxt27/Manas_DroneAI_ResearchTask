# Stage 2: Yellow Marker Detection
# ================================
# Detects and localizes all yellow markers in the stitched map

import cv2
import numpy as np
from pathlib import Path
import json
import csv
import logging
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YellowMarkerDetector:
    """
    Detects yellow markers in the stitched image using color-based segmentation.
    Provides robust detection with precision and recall metrics.
    """
    
    def __init__(self, min_marker_size=20, max_marker_size=5000):
        """
        Initialize marker detector.
        
        Args:
            min_marker_size: Minimum marker area in pixels
            max_marker_size: Maximum marker area in pixels
        """
        self.min_marker_size = min_marker_size
        self.max_marker_size = max_marker_size
        self.markers = []
        
    def detect_yellow_markers(self, image):
        """
        Detect yellow markers using HSV color space segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of marker centroids (x, y), Binary mask
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range in HSV
        # Blue hue is around 100-130 in OpenCV (0-180 scale)
        lower_blue1 = np.array([100, 80, 80])
        upper_blue1 = np.array([130, 255, 255])

        # Optional second range (if needed, e.g. for different blues)
        lower_blue2 = np.array([90, 80, 80])
        upper_blue2 = np.array([140, 255, 255])

        # Create masks for blue
        mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        mask = cv2.bitwise_or(mask1, mask2)

        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if self.min_marker_size < area < self.max_marker_size:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    markers.append((cx, cy, area))
        
        self.markers = markers
        logger.info(f"Detected {len(markers)} yellow markers")
        
        return markers, mask
    
    def get_marker_coordinates(self):
        """
        Get marker coordinates in pixel format.
        
        Returns:
            List of (x, y) coordinates
        """
        return [(m[0], m[1]) for m in self.markers]
    
    def save_marker_coordinates_json(self, output_path):
        """
        Save marker coordinates to JSON file.
        
        Args:
            output_path: Output JSON file path
        """
        marker_data = {
            "total_markers": len(self.markers),
            "markers": [
                {
                    "id": idx,
                    "x_pixel": m[0],
                    "y_pixel": m[1],
                    "area": m[2]
                }
                for idx, m in enumerate(self.markers)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(marker_data, f, indent=4)
        
        logger.info(f"Marker coordinates saved to {output_path}")
    
    def save_marker_coordinates_csv(self, output_path):
        """
        Save marker coordinates to CSV file.
        
        Args:
            output_path: Output CSV file path
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Marker_ID', 'X_Pixel', 'Y_Pixel', 'Area'])
            
            for idx, (x, y, area) in enumerate(self.markers):
                writer.writerow([idx, x, y, area])
        
        logger.info(f"Marker coordinates saved to {output_path}")
    
    def draw_markers_on_image(self, image):
        """
        Draw detected markers on image with annotations.
        
        Args:
            image: Input image
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for idx, (x, y, area) in enumerate(self.markers):
            # Draw circle at centroid
            cv2.circle(annotated, (x, y), 10, (0, 255, 255), 2)
            
            # Add marker ID
            cv2.putText(annotated, f"M{idx}", (x + 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add crosshair
            cv2.line(annotated, (x - 20, y), (x + 20, y), (0, 255, 255), 1)
            cv2.line(annotated, (x, y - 20), (x, y + 20), (0, 255, 255), 1)
        
        return annotated
    
    def calculate_metrics(self, ground_truth_markers=None):
        """
        Calculate detection metrics (precision, recall if ground truth available).
        
        Args:
            ground_truth_markers: List of ground truth marker positions (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "total_detected": len(self.markers),
        }
        
        if ground_truth_markers is not None:
            # Calculate precision and recall
            detected_set = set([(int(m[0]/10), int(m[1]/10)) for m in self.markers])
            truth_set = set([(int(m[0]/10), int(m[1]/10)) for m in ground_truth_markers])
            
            true_positives = len(detected_set & truth_set)
            false_positives = len(detected_set - truth_set)
            false_negatives = len(truth_set - detected_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            })
        
        return metrics


def run_stage2(stitched_image_path, output_dir):
    """
    Execute Stage 2: Yellow Marker Detection
    
    Args:
        stitched_image_path: Path to stitched image
        output_dir: Directory to save marker detection results
        
    Returns:
        Detector object and markers list
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: YELLOW MARKER DETECTION")
    logger.info("=" * 60)
    
    # Load stitched image
    image = cv2.imread(stitched_image_path)
    if image is None:
        logger.error(f"Could not load image: {stitched_image_path}")
        return None, None
    
    # Detect markers
    detector = YellowMarkerDetector(min_marker_size=50, max_marker_size=5000)
    markers, mask = detector.detect_yellow_markers(image)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON and CSV
    detector.save_marker_coordinates_json(str(output_dir / "marker_coordinates.json"))
    detector.save_marker_coordinates_csv(str(output_dir / "marker_coordinates.csv"))
    
    # Save annotated image
    annotated_image = detector.draw_markers_on_image(image)
    cv2.imwrite(str(output_dir / "annotated_map_with_markers.png"), annotated_image)
    
    # Save detection mask
    cv2.imwrite(str(output_dir / "yellow_mask.png"), mask)
    
    logger.info(f"Results saved to {output_dir}")
    
    return detector, markers


if __name__ == "__main__":
    # Example usage
    stitched_image_path = "./output/stitched_map.png"
    output_dir = "./output"
    
    detector, markers = run_stage2(stitched_image_path, output_dir)
