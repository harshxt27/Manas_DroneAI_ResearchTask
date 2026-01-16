# Project MANAS - Drone Image Processing Pipeline
## Complete Implementation Guide

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [File Structure](#file-structure)
4. [Running the Pipeline](#running-the-pipeline)
5. [Detailed Stage Descriptions](#detailed-stage-descriptions)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## 🎯 Project Overview

This project implements a complete drone image processing pipeline with three interconnected stages:

### **Stage 1: Image Stitching**
- Combines multiple overlapping PNG drone images into a single coherent orthomosaic map
- Uses SIFT feature detection and homography transformation
- Handles image alignment with rotation and perspective correction

### **Stage 2: Yellow Marker Detection**
- Detects and localizes yellow circular markers in the stitched map
- Uses HSV color space segmentation for robust detection
- Provides precision/recall metrics and outputs marker coordinates

### **Stage 3: Costmap & Path Planning**
- Generates a traversability costmap from terrain features
- Plans optimal paths between markers using A* algorithm
- Minimizes cumulative cost for most traversable routes

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

**Required packages:**
- opencv-python (image processing)
- numpy (numerical operations)
- scikit-image (advanced image operations)
- scipy (scientific computing)
- Pillow (image I/O)
- matplotlib (visualization)
- tqdm (progress bars)

### Step 2: Prepare Dataset

1. Place all PNG drone images in the `./dataset` folder
2. Images should have overlapping regions (minimum 20-30% overlap recommended)
3. Images are processed alphabetically by filename

**Example dataset structure:**
```
dataset/
├── image_001.png
├── image_002.png
├── image_003.png
├── image_004.png
└── image_005.png
```

### Step 3: Create Output Directory

```bash
mkdir output
```

---

## 📁 File Structure

```
project_root/
├── code/
│   ├── main.py                      # Main pipeline orchestrator
│   ├── image_stitching.py           # Stage 1: Image stitching
│   ├── marker_detection.py          # Stage 2: Marker detection
│   ├── costmap_pathplanning.py      # Stage 3: Costmap & path planning
│   └── utilities.py                 # Helper functions (optional)
│
├── dataset/                         # Input PNG images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
├── output/                          # Generated results
│   ├── stage1_stitched_map.png
│   ├── annotated_map_with_markers.png
│   ├── costmap_visualization.png
│   ├── map_with_planned_paths.png
│   ├── marker_coordinates.json
│   ├── marker_coordinates.csv
│   ├── planned_paths.json
│   ├── pipeline.log
│   └── pipeline_report.txt
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore file
```

---

## ⚡ Running the Pipeline

### Quick Start

Run the complete pipeline:

```bash
cd code
python main.py
```

The pipeline will:
1. Load all PNG images from `./dataset`
2. Stitch images together into orthomosaic
3. Detect yellow markers
4. Generate costmap
5. Plan paths between markers
6. Save all results to `./output`

### Expected Output Messages

```
================================================================================
PROJECT MANAS - DRONE IMAGE PROCESSING PIPELINE
================================================================================

2026-01-16 12:30:45 - __main__ - INFO - ============================================================
2026-01-16 12:30:45 - __main__ - INFO - STAGE 1: IMAGE STITCHING
2026-01-16 12:30:45 - __main__ - INFO - ============================================================
Loading images: 100%|████████| 5/5 [00:02<00:00, 2.50 img/s]
2026-01-16 12:30:47 - __main__ - INFO - Found 5 images to process
2026-01-16 12:30:47 - __main__ - INFO - Stitching 5 images...
[... stitching progress ...]
2026-01-16 12:31:15 - __main__ - INFO - Stitched image dimensions: 2560x1920 pixels
2026-01-16 12:31:15 - __main__ - INFO - ✓ Stage 1 completed successfully

[... Stage 2 and Stage 3 output ...]

================================================================================
PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
All outputs saved to: ./output
```

---

## 📊 Detailed Stage Descriptions

### Stage 1: Image Stitching (`image_stitching.py`)

**Algorithm Flow:**
1. Load all PNG images from dataset folder
2. For each pair of consecutive images:
   - Detect SIFT keypoints and descriptors
   - Match features using BF matcher
   - Apply Lowe's ratio test to filter ambiguous matches
   - Calculate homography matrix using RANSAC
   - Warp first image to align with second
3. Blend overlapping regions
4. Crop black borders from result

**Key Parameters:**
```python
min_matches = 10              # Minimum features to match
SIFT_threshold = 0.75         # Lowe's ratio test threshold
```

**Output:**
- `stage1_stitched_map.png`: High-resolution orthomosaic

---

### Stage 2: Yellow Marker Detection (`marker_detection.py`)

**Algorithm Flow:**
1. Convert image to HSV color space
2. Define yellow color ranges:
   - Primary: HSV(15-35, 100-255, 100-255)
   - Secondary: HSV(0-15, 100-255, 100-255)
3. Create binary mask of yellow regions
4. Apply morphological operations (closing + opening)
5. Find contours in mask
6. Filter by size constraints
7. Calculate centroid of each valid contour

**Key Parameters:**
```python
min_marker_size = 50 px²      # Minimum marker area
max_marker_size = 5000 px²    # Maximum marker area
```

**Outputs:**
- `marker_coordinates.json`: Marker positions in JSON format
- `marker_coordinates.csv`: Marker positions in CSV format
- `annotated_map_with_markers.png`: Map with marker annotations
- `yellow_mask.png`: Binary mask of detected markers

---

### Stage 3: Costmap & Path Planning (`costmap_pathplanning.py`)

**Costmap Generation:**
1. Convert image to grayscale
2. Detect terrain edges using Canny edge detection
3. Dilate edges to create obstacle zones
4. Apply distance transform (higher distance = more traversable)
5. Normalize to 0-255 range
6. Apply Gaussian blur for smoothness

**Path Planning (A* Algorithm):**
1. Create priority queue with start position
2. Expand neighbors with lowest f-score (g + h)
3. Use Manhattan distance as heuristic
4. Consider terrain cost at each step
5. Reconstruct path when goal reached

**Outputs:**
- `costmap_visualization.png`: Heatmap of traversability (red=obstacle, green=traversable)
- `planned_paths.json`: Waypoints for each path with metrics
- `map_with_planned_paths.png`: Visual representation of planned routes

---

## 📤 Output Files

### JSON Files

**marker_coordinates.json:**
```json
{
    "total_markers": 5,
    "markers": [
        {"id": 0, "x_pixel": 250, "y_pixel": 180, "area": 425},
        {"id": 1, "x_pixel": 680, "y_pixel": 350, "area": 412},
        ...
    ]
}
```

**planned_paths.json:**
```json
[
    {
        "from_marker": 0,
        "to_marker": 1,
        "waypoints": [[250, 180], [260, 185], ...],
        "metrics": {
            "path_length": 450.5,
            "average_cost": 128.3,
            "total_waypoints": 47,
            "estimated_traversal_time_sec": 901.0
        }
    }
]
```

### CSV Files

**marker_coordinates.csv:**
```
Marker_ID,X_Pixel,Y_Pixel,Area
0,250,180,425
1,680,350,412
2,920,210,438
...
```

### Image Files

- **stage1_stitched_map.png**: Final stitched orthomosaic
- **annotated_map_with_markers.png**: Markers highlighted with IDs
- **costmap_visualization.png**: Traversability heatmap
- **map_with_planned_paths.png**: Paths overlaid on stitched map

### Log Files

- **pipeline.log**: Detailed execution log
- **pipeline_report.txt**: Human-readable summary report

---

## 🔧 Troubleshooting

### Issue 1: "No PNG images found"
**Solution:**
- Check that PNG files exist in `./dataset` folder
- Verify file extensions are exactly `.png` (lowercase)
- Ensure read permissions on files

### Issue 2: "Not enough matches" warning
**Possible causes:**
- Low image overlap (< 20%)
- Images too dissimilar or from different angles
- Poor image quality or lighting variations

**Solutions:**
- Verify images have sufficient overlap
- Ensure images are from similar altitude
- Check image resolution is consistent
- Reduce `min_matches` threshold if necessary

### Issue 3: No markers detected
**Possible causes:**
- Yellow color range doesn't match actual marker color
- Markers are too small/large for size constraints
- Lighting conditions affect color detection

**Solutions:**
- Adjust HSV color ranges in `marker_detection.py`
- Modify `min_marker_size` and `max_marker_size` parameters
- Check image lighting and contrast

### Issue 4: Path planning fails
**Possible causes:**
- Insufficient markers (need at least 2)
- Markers too close to image edges
- Costmap blocked between markers

**Solutions:**
- Ensure markers are detected successfully in Stage 2
- Check marker coordinates in JSON output
- Visualize costmap to identify blocked regions

### Issue 5: Memory error with large images
**Solutions:**
```python
# Resize images before processing
scale = 0.5  # 50% of original size
img_resized = cv2.resize(img, None, fx=scale, fy=scale)
```

### Issue 6: Slow performance
**Optimization tips:**
- Reduce image resolution
- Decrease SIFT match threshold
- Simplify costmap grid size
- Run on GPU if available

---

## ⚙️ Performance Optimization

### For Large Datasets

1. **Resize Images Before Stitching:**
```python
scale = 0.7
resized_imgs = [cv2.resize(img, None, fx=scale, fy=scale) for img in images]
```

2. **Reduce Feature Matches:**
```python
min_matches = 5  # Lower threshold
```

3. **Simplify Costmap:**
```python
costmap_gen = CostmapGenerator(grid_size=100)  # Larger grid cells
```

### For Faster Path Planning

1. **Dijkstra instead of A*:**
- Less overhead for small maps
- Suitable when heuristic isn't available

2. **Bidirectional Search:**
- Search from both start and goal simultaneously
- Reduces search space

### For Better Accuracy

1. **Increase SIFT matches:**
```python
min_matches = 20
```

2. **Finer grid cells:**
```python
costmap_gen = CostmapGenerator(grid_size=25)
```

3. **Multiple path planning attempts:**
- Try different heuristics
- Compare results

---

## 📧 Submission Checklist

Before submitting to projectmanas.mit@gmail.com:

- [ ] All three stages execute successfully
- [ ] All output files generated in `./output` folder
- [ ] Code is well-documented with comments
- [ ] README.md contains setup and execution instructions
- [ ] requirements.txt has all dependencies
- [ ] Git repository link ready
- [ ] Performance metrics documented
- [ ] Challenges and solutions documented

**Submission Email Template:**
```
Subject: Project MANAS Task 2 - Drone Image Processing

Body:
Name: [Your Name]
Reg No: [Registration Number]
Branch: [Your Branch]
Mobile No: [Your Mobile Number]

Git Repository: [GitHub Link]
```

---

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- SIFT Feature Detection: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
- A* Pathfinding: https://en.wikipedia.org/wiki/A*_search_algorithm
- HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV

---

## 📝 License

This project is for educational purposes under Project MANAS.

---

## ❓ FAQ

**Q: Can I use different image formats?**
A: Modify the glob pattern in `image_stitching.py`:
```python
image_files = sorted(Path(dataset_path).glob('*.jpg'))  # For JPG files
```

**Q: How do I adjust marker detection sensitivity?**
A: Edit HSV ranges in `marker_detection.py`:
```python
lower_yellow1 = np.array([10, 80, 80])   # More sensitive
upper_yellow1 = np.array([40, 255, 255])
```

**Q: Can I use my own path planning algorithm?**
A: Yes! Replace the A* implementation in `costmap_pathplanning.py` with your algorithm.

**Q: What if I have thousands of images?**
A: Implement incremental stitching or use specialized stitching libraries like OpenCV's Stitcher class.

---

## 🎓 Learning Outcomes

By completing this project, you will understand:
- Image feature detection and matching
- Homography transformation and image registration
- Color space conversions and object detection
- Grid-based path planning algorithms
- Pipeline orchestration and error handling
- Data visualization and reporting

---

**Good luck with Project MANAS!** 🚀
