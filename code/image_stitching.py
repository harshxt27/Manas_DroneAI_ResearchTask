# Stage 1: Image Stitching
# ========================
# Combines overlapping drone PNG images into a single orthomosaic map

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageStitcher:
    """
    Stitches multiple overlapping drone images into a single coherent map.
    Uses feature matching and homography for alignment.
    """
    
    def __init__(self, min_matches=10):
        """
        Initialize the image stitcher.
        
        Args:
            min_matches: Minimum number of feature matches required for stitching
        """
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def load_images(self, dataset_path):
        """
        Load all PNG images from dataset folder and sort them.
        
        Args:
            dataset_path: Path to folder containing PNG images
            
        Returns:
            List of (image, filename) tuples
        """
        images = []
        image_files = sorted(Path(dataset_path).glob('*.png'))
        
        if not image_files:
            logger.error(f"No PNG images found in {dataset_path}")
            return images
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc="Loading images"):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img, img_path.name))
            else:
                logger.warning(f"Failed to load {img_path}")
        
        return images
    
    def detect_and_compute_features(self, image):
        """
        Detect SIFT keypoints and compute descriptors.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def find_matches(self, desc1, desc2):
        """
        Find feature matches between two descriptors using Lowe's ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Lowe's ratio test - discard ambiguous matches
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography(self, kp1, kp2, matches):
        """
        Calculate homography matrix from matched keypoints.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Matched keypoints
            
        Returns:
            Homography matrix or None if insufficient matches
        """
        if len(matches) < self.min_matches:
            logger.warning(f"Not enough matches ({len(matches)}) - need at least {self.min_matches}")
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def stitch_pair(self, img1, img2):
        """
        Stitch two images together.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Stitched image or None if stitching fails
        """
        kp1, desc1 = self.detect_and_compute_features(img1)
        kp2, desc2 = self.detect_and_compute_features(img2)
        
        if desc1 is None or desc2 is None:
            logger.warning("Could not compute features for one or both images")
            return None
        
        matches = self.find_matches(desc1, desc2)
        H = self.find_homography(kp1, kp2, matches)
        
        if H is None:
            logger.warning("Could not compute homography matrix")
            return None
        
        # Warp img1 to align with img2
        h, w = img2.shape[:2]
        img1_warped = cv2.warpPerspective(img1, H, (w * 2, h * 2))
        
        # Blend the images
        result = img1_warped.copy()
        result[0:h, 0:w] = img2
        
        return result
    
    def stitch_all_images(self, images):
        """
        Stitch all images sequentially into one orthomosaic.
        
        Args:
            images: List of (image, filename) tuples
            
        Returns:
            Final stitched image
        """
        if not images:
            logger.error("No images to stitch")
            return None
        
        result = images[0][0].copy()
        logger.info(f"Starting with image: {images[0][1]}")
        
        for i in range(1, len(images)):
            logger.info(f"Stitching image {i+1}/{len(images)}: {images[i][1]}")
            stitched = self.stitch_pair(result, images[i][0])
            
            if stitched is not None:
                result = stitched
            else:
                logger.warning(f"Could not stitch image {i}, skipping")
        
        # Crop unnecessary borders
        result = self.crop_black_borders(result)
        
        return result
    
    def crop_black_borders(self, image):
        """
        Remove black borders from stitched image.
        
        Args:
            image: Input image
            
        Returns:
            Cropped image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return image[y:y+h, x:x+w]
        
        return image
    
    def save_stitched_image(self, image, output_path):
        """
        Save the stitched image to file.
        
        Args:
            image: Stitched image
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
        logger.info(f"Stitched image saved to {output_path}")
        
        # Print image dimensions
        h, w = image.shape[:2]
        logger.info(f"Stitched image dimensions: {w}x{h} pixels")


def run_stage1(dataset_path, output_path):
    """
    Execute Stage 1: Image Stitching
    
    Args:
        dataset_path: Path to PNG images
        output_path: Path to save stitched image
        
    Returns:
        Stitched image and path
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: IMAGE STITCHING")
    logger.info("=" * 60)
    
    stitcher = ImageStitcher(min_matches=10)
    images = stitcher.load_images(dataset_path)
    
    if not images:
        logger.error("No images loaded")
        return None, None
    
    logger.info(f"Stitching {len(images)} images...")
    stitched_image = stitcher.stitch_all_images(images)
    
    if stitched_image is not None:
        stitcher.save_stitched_image(stitched_image, output_path)
        return stitched_image, output_path
    else:
        logger.error("Image stitching failed")
        return None, None


if __name__ == "__main__":
    # Example usage
    dataset_path = "./dataset"
    output_path = "./output/stitched_map.png"
    
    stitched_img, path = run_stage1(dataset_path, output_path)
