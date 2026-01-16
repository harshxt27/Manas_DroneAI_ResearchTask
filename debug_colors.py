# Debug Script - Find What Colors Are in Your Image
# Run this to discover marker colors

import cv2
import numpy as np
from pathlib import Path

def analyze_image_colors(image_path):
    """
    Analyze the stitched image to find marker colors
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print("\n" + "="*60)
    print("IMAGE COLOR ANALYSIS")
    print("="*60)
    print(f"Image size: {img.shape}")
    
    # Try different color ranges to find what's in the image
    print("\nTesting color ranges:\n")
    
    # Yellow range
    lower = np.array([15, 100, 100])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    yellow_pixels = cv2.countNonZero(mask)
    print(f"YELLOW (H:15-35):       {yellow_pixels:,} pixels")
    
    # Orange-Yellow
    lower = np.array([5, 100, 100])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    orange_pixels = cv2.countNonZero(mask)
    print(f"ORANGE-YELLOW (H:5-25): {orange_pixels:,} pixels")
    
    # Red
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    red_pixels = cv2.countNonZero(mask)
    print(f"RED (H:0-10):           {red_pixels:,} pixels")
    
    # White
    lower = np.array([0, 0, 200])
    upper = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower, upper)
    white_pixels = cv2.countNonZero(mask)
    print(f"WHITE (V:200-255):      {white_pixels:,} pixels")
    
    # Green
    lower = np.array([40, 100, 100])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_pixels = cv2.countNonZero(mask)
    print(f"GREEN (H:40-80):        {green_pixels:,} pixels")
    
    # Blue
    lower = np.array([100, 100, 100])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    blue_pixels = cv2.countNonZero(mask)
    print(f"BLUE (H:100-130):       {blue_pixels:,} pixels")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    color_counts = {
        "YELLOW": yellow_pixels,
        "ORANGE-YELLOW": orange_pixels,
        "RED": red_pixels,
        "WHITE": white_pixels,
        "GREEN": green_pixels,
        "BLUE": blue_pixels
    }
    
    max_color = max(color_counts, key=color_counts.get)
    max_pixels = color_counts[max_color]
    
    print(f"\nHighest count: {max_color} with {max_pixels:,} pixels")
    
    if max_pixels == 0:
        print("\n⚠️  WARNING: No clear color markers detected!")
        print("Markers might be:")
        print("  - Blended with background")
        print("  - Too small or too large")
        print("  - Different color than expected")
        print("  - Low contrast with terrain")
    else:
        print(f"\n✓ Found {max_color} markers in image")
        print("  Update marker_detection.py HSV ranges accordingly")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    image_path = "./output/stage1_stitched_map.png"
    analyze_image_colors(image_path)
