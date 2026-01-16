# Utility Functions
# =================
# Helper functions for visualization, debugging, and data processing

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_comparison_image(image1, image2, title1="Image 1", title2="Image 2"):
    """
    Create side-by-side comparison of two images.
    
    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
    
    Returns:
        Comparison image
    """
    # Resize to same height if different
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    max_height = max(h1, h2)
    
    if h1 < max_height:
        image1 = cv2.copyMakeBorder(image1, (max_height - h1) // 2, (max_height - h1) // 2, 
                                     0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if h2 < max_height:
        image2 = cv2.copyMakeBorder(image2, (max_height - h2) // 2, (max_height - h2) // 2, 
                                     0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    comparison = np.hstack([image1, image2])
    
    # Add titles
    h, w = comparison.shape[:2]
    title_img = np.ones((80, w, 3), dtype=np.uint8) * 255
    cv2.putText(title_img, title1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(title_img, title2, (w // 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    comparison = np.vstack([title_img, comparison])
    
    return comparison


def visualize_pipeline_results(output_dir):
    """
    Create comprehensive visualization of all pipeline results.
    
    Args:
        output_dir: Directory containing all output files
    """
    output_dir = Path(output_dir)
    
    logger.info("Creating comprehensive visualization...")
    
    # Load results
    files = {
        'stitched': output_dir / "stage1_stitched_map.png",
        'marked': output_dir / "annotated_map_with_markers.png",
        'costmap': output_dir / "costmap_visualization.png",
        'paths': output_dir / "map_with_planned_paths.png"
    }
    
    images = {}
    for name, path in files.items():
        if path.exists():
            images[name] = cv2.imread(str(path))
            logger.info(f"Loaded {name}")
        else:
            logger.warning(f"Not found: {path}")
    
    if not images:
        logger.error("No output images found")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Project MANAS - Complete Pipeline Results', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for name, img in images.items():
        row = plot_idx // 2
        col = plot_idx % 2
        
        ax = axes[row, col]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx // 2, idx % 2].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "pipeline_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    return fig


def generate_heatmap(data, title="Heatmap", output_path=None):
    """
    Generate heatmap visualization.
    
    Args:
        data: 2D numpy array
        title: Heatmap title
        output_path: Optional output file path
    
    Returns:
        Heatmap image
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cost Value')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Heatmap saved to {output_path}")
    
    return plt.gcf()


def print_stage_summary(stage_num, status, details):
    """
    Print formatted stage summary.
    
    Args:
        stage_num: Stage number (1, 2, 3)
        status: 'SUCCESS' or 'FAILED'
        details: Dictionary of stage-specific details
    """
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num} SUMMARY")
    print("=" * 70)
    print(f"Status: {status}")
    
    for key, value in details.items():
        print(f"  {key}: {value}")
    
    print("=" * 70 + "\n")


def validate_image(image_path):
    """
    Validate if image file is readable and valid.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    img = cv2.imread(str(image_path))
    
    if img is None:
        logger.error(f"Invalid image: {image_path}")
        return False
    
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        logger.error(f"Image too small: {w}x{h}")
        return False
    
    logger.info(f"Valid image: {image_path} ({w}x{h})")
    return True


def resize_image_to_memory_limit(image, max_memory_mb=500):
    """
    Resize image if it exceeds memory limit.
    
    Args:
        image: Input image
        max_memory_mb: Maximum allowed memory in MB
    
    Returns:
        Resized image if necessary
    """
    h, w, c = image.shape
    actual_size_mb = (h * w * c) / (1024 * 1024)
    
    if actual_size_mb > max_memory_mb:
        scale = np.sqrt(max_memory_mb / actual_size_mb)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        logger.warning(f"Image resized to {new_w}x{new_h}")
    
    return image


def plot_markers_distribution(markers, image_shape, output_path=None):
    """
    Plot distribution of detected markers.
    
    Args:
        markers: List of (x, y, area) tuples
        image_shape: Shape of image (height, width)
        output_path: Optional output file path
    
    Returns:
        Figure object
    """
    if not markers:
        logger.warning("No markers to plot")
        return None
    
    xs = [m[0] for m in markers]
    ys = [m[1] for m in markers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # X distribution
    ax1.hist(xs, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Marker X Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Y distribution
    ax2.hist(ys, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Y Position (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Marker Y Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Marker Distribution ({len(markers)} markers)', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {output_path}")
    
    return fig


def estimate_processing_time(num_images, avg_time_per_image=5):
    """
    Estimate total processing time.
    
    Args:
        num_images: Number of images to stitch
        avg_time_per_image: Average seconds per image pair
    
    Returns:
        Estimated time in minutes
    """
    total_seconds = (num_images - 1) * avg_time_per_image
    minutes = total_seconds / 60
    return minutes


def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    output_dir = "./output"
    visualize_pipeline_results(output_dir)
