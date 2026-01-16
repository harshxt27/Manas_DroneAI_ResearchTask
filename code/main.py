# Main Pipeline Orchestrator
# ==========================
# Runs all three stages of the Project MANAS drone imagery processing pipeline

import sys
import logging
from pathlib import Path
import cv2

# Import stage modules
from image_stitching import run_stage1
from marker_detection import run_stage2
from costmap_pathplanning import run_stage3


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./output/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DroneImagePipeline:
    """
    Complete pipeline for drone imagery processing:
    1. Image stitching
    2. Blue marker detection
    3. Costmap generation and path planning
    """
    
    def __init__(self, dataset_path, output_path):
        """
        Initialize pipeline.
        
        Args:
            dataset_path: Path to PNG image dataset
            output_path: Path for output results
        """
        self.dataset_path = dataset_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.stitched_image = None
        self.stitched_image_path = None
        self.markers = None
        self.detector = None
        self.stage3_planner = None
        
    def run_all_stages(self):
        """Execute the complete pipeline."""
        
        logger.info("\n" + "=" * 80)
        logger.info("PROJECT MANAS - DRONE IMAGE PROCESSING PIPELINE")
        logger.info("=" * 80 + "\n")
        
        try:
            # Stage 1: Image Stitching
            if not self.run_stage_1():
                logger.error("Stage 1 failed - aborting pipeline")
                return False
            
            logger.info("=" * 80)
            
            # Stage 2: Marker Detection
            if not self.run_stage_2():
                logger.error("Stage 2 failed - aborting pipeline")
                return False
            
            logger.info("=" * 80)
            
            # Stage 3: Costmap and Path Planning
            if not self.run_stage_3():
                logger.error("Stage 3 failed - continuing to report generation")
            
            # Generate report
            self.generate_report()
            
            # Print summary
            self.print_summary()
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"All outputs saved to: {self.output_path}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return False
    
    def run_stage_1(self):
        """Execute Stage 1: Image Stitching"""
        try:
            logger.info("=" * 80)
            output_file = str(self.output_path / "stage1_stitched_map.png")
            self.stitched_image, self.stitched_image_path = run_stage1(
                self.dataset_path, 
                output_file
            )
            
            if self.stitched_image is not None:
                logger.info("✓ Stage 1 completed successfully")
                return True
            else:
                logger.error("✗ Stage 1 failed")
                return False
                
        except Exception as e:
            logger.error(f"Stage 1 error: {str(e)}", exc_info=True)
            return False
    
    def run_stage_2(self):
        """Execute Stage 2: Blue Marker Detection"""
        try:
            logger.info("=" * 80)
            
            if self.stitched_image_path is None:
                logger.error("Cannot run Stage 2 - no stitched image")
                return False
            
            self.detector, self.markers = run_stage2(
                self.stitched_image_path,
                str(self.output_path)
            )
            
            if self.markers is not None and len(self.markers) > 0:
                logger.info("✓ Stage 2 completed successfully")
                return True
            else:
                logger.warning("✗ Stage 2 completed but no markers detected")
                self.markers = []
                return True  # Don't abort if no markers found
                
        except Exception as e:
            logger.error(f"Stage 2 error: {str(e)}", exc_info=True)
            return False
    
    def run_stage_3(self):
        """Execute Stage 3: Costmap and Path Planning"""
        try:
            logger.info("=" * 80)
            
            marker_coords_json = str(self.output_path / "marker_coordinates.json")
            
            self.stage3_planner = run_stage3(
                self.stitched_image_path,
                marker_coords_json,
                str(self.output_path)
            )
            
            if self.stage3_planner is not None:
                logger.info("✓ Stage 3 completed successfully")
                return True
            else:
                logger.error("✗ Stage 3 failed")
                return False
                
        except Exception as e:
            logger.error(f"Stage 3 error: {str(e)}", exc_info=True)
            return False
    
    def generate_report(self):
        """Generate comprehensive pipeline report with proper Unicode encoding."""
        
        logger.info("\nGenerating comprehensive report...")
        
        report_path = self.output_path / "pipeline_report.txt"
        
        try:
            # Use UTF-8 encoding to handle special characters
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("PROJECT MANAS - DRONE IMAGE PROCESSING PIPELINE REPORT\n")
                f.write("=" * 70 + "\n\n")
                
                # Stage 1 Report
                f.write("STAGE 1: IMAGE STITCHING\n")
                f.write("-" * 70 + "\n")
                if self.stitched_image is not None:
                    h, w = self.stitched_image.shape[:2]
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"Stitched Image Dimensions: {w}x{h} pixels\n")
                    f.write(f"Output: stage1_stitched_map.png\n")
                else:
                    f.write(f"Status: FAILED\n")
                f.write("\n")
                
                # Stage 2 Report
                f.write("STAGE 2: BLUE MARKER DETECTION\n")
                f.write("-" * 70 + "\n")
                if self.detector and self.markers and len(self.markers) > 0:
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"Total Markers Detected: {len(self.markers)}\n")
                    f.write(f"\nMarker Details:\n")
                    for idx, (x, y, area) in enumerate(self.markers):
                        f.write(f"  Marker {idx}: Position ({x}, {y}), Area {area:.1f} px2\n")
                    f.write(f"\nOutputs:\n")
                    f.write(f"  - marker_coordinates.json\n")
                    f.write(f"  - marker_coordinates.csv\n")
                    f.write(f"  - annotated_map_with_markers.png\n")
                    f.write(f"  - yellow_mask.png\n")
                else:
                    f.write(f"Status: NO MARKERS DETECTED\n")
                    f.write(f"The pipeline completed but found no markers in the image.\n")
                f.write("\n")
                
                # Stage 3 Report
                f.write("STAGE 3: COSTMAP & PATH PLANNING\n")
                f.write("-" * 70 + "\n")
                if self.stage3_planner:
                    if self.stage3_planner.paths and len(self.stage3_planner.paths) > 0:
                        f.write(f"Status: SUCCESS\n")
                        f.write(f"Total Paths Planned: {len(self.stage3_planner.paths)}\n")
                        f.write(f"\nPath Details:\n")
                        
                        for path in self.stage3_planner.paths:
                            metrics = self.stage3_planner.calculate_path_metrics(path)
                            # Use ASCII arrow instead of Unicode
                            f.write(f"\n  Path {path['from_marker']} -> {path['to_marker']}:\n")
                            f.write(f"    - Length: {metrics['path_length']:.2f} pixels\n")
                            f.write(f"    - Average Cost: {metrics['average_cost']:.2f}\n")
                            f.write(f"    - Total Waypoints: {metrics['total_waypoints']}\n")
                            f.write(f"    - Est. Traversal Time: {metrics['estimated_traversal_time_sec']:.2f} seconds\n")
                        
                        f.write(f"\nOutputs:\n")
                        f.write(f"  - costmap_visualization.png\n")
                        f.write(f"  - planned_paths.json\n")
                        f.write(f"  - map_with_planned_paths.png\n")
                    else:
                        f.write(f"Status: COSTMAP GENERATED (No paths planned)\n")
                        f.write(f"Costmap created but insufficient markers for path planning.\n")
                        f.write(f"\nOutputs:\n")
                        f.write(f"  - costmap_visualization.png\n")
                else:
                    f.write(f"Status: FAILED\n")
                f.write("\n")
                
                # Summary
                f.write("=" * 70 + "\n")
                f.write("PIPELINE SUMMARY\n")
                f.write("=" * 70 + "\n")
                f.write(f"Dataset Path: {self.dataset_path}\n")
                f.write(f"Output Directory: {self.output_path}\n")
                f.write(f"Log File: {self.output_path / 'pipeline.log'}\n")
                f.write(f"\nAll outputs successfully generated!\n")
            
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        
        if self.stitched_image is not None:
            h, w = self.stitched_image.shape[:2]
            print(f"✓ Stage 1: Image Stitched ({w}x{h} pixels)")
        else:
            print("✗ Stage 1: Failed")
        
        if self.markers and len(self.markers) > 0:
            print(f"✓ Stage 2: {len(self.markers)} markers detected")
        else:
            print("✗ Stage 2: No markers detected")
        
        if self.stage3_planner:
            if self.stage3_planner.paths and len(self.stage3_planner.paths) > 0:
                print(f"✓ Stage 3: {len(self.stage3_planner.paths)} paths planned")
            else:
                print("✓ Stage 3: Costmap generated (no paths)")
        else:
            print("✗ Stage 3: Failed")
        
        print("\n" + "=" * 80)
        print(f"Output directory: {self.output_path}")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    
    # Configuration
    dataset_path = "./dataset"
    output_path = "./output"
    
    # Verify dataset exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Create and run pipeline
    pipeline = DroneImagePipeline(dataset_path, output_path)
    success = pipeline.run_all_stages()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)