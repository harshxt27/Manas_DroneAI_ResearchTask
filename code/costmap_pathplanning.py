# Stage 3: Costmap Generation and Path Planning
# =============================================
# Creates traversability costmap and plans optimal paths between markers

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from scipy import ndimage
from collections import deque
import heapq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CostmapGenerator:
    """
    Generates a traversability costmap from terrain features.
    Lower cost = more traversable.
    """
    
    def __init__(self, grid_size=50):
        """
        Initialize costmap generator.
        
        Args:
            grid_size: Size of each grid cell in pixels
        """
        self.grid_size = grid_size
        self.costmap = None
        
    def generate_costmap(self, image):
        """
        Generate costmap from image using terrain analysis.
        
        Args:
            image: Input stitched image
            
        Returns:
            Costmap (0-255 normalized)
        """
        logger.info("Generating traversability costmap...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect terrain features (edges indicate obstacles)
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges to create obstacle zones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        obstacles = cv2.dilate(edges, kernel, iterations=2)
        
        # Distance transform - further from obstacles = lower cost
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(obstacles), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Normalize distance to 0-255
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Invert: higher distance (traversable) = lower cost
        costmap = 255 - dist_norm
        
        # Apply Gaussian blur for smoothness
        costmap = cv2.GaussianBlur(costmap, (21, 21), 0)
        
        self.costmap = costmap
        logger.info(f"Costmap generated: shape {costmap.shape}")
        
        return costmap
    
    def save_costmap(self, output_path):
        """Save costmap as image with color visualization."""
        if self.costmap is None:
            logger.error("No costmap generated")
            return
        
        # Apply colormap for visualization
        costmap_color = cv2.applyColorMap(self.costmap, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, costmap_color)
        logger.info(f"Costmap saved to {output_path}")
        
        return costmap_color


class PathPlanner:
    """
    Plans optimal paths using A* algorithm based on costmap.
    """
    
    def __init__(self, costmap, grid_size=50):
        """
        Initialize path planner.
        
        Args:
            costmap: Traversability costmap
            grid_size: Size of each grid cell
        """
        self.costmap = costmap
        self.grid_size = grid_size
        self.height, self.width = costmap.shape
        self.paths = []
        
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic for A*."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_neighbors(self, pos):
        """Get valid 8-connected neighbors."""
        neighbors = []
        x, y = pos
        
        # 8-directional movement (including diagonals)
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def a_star_search(self, start, goal):
        """
        A* pathfinding algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of waypoints from start to goal
        """
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Cost based on terrain traversability
                cost = self.costmap[neighbor[0], neighbor[1]] / 255.0
                tentative_g = g_score[current] + cost + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        logger.warning(f"No path found from {start} to {goal}")
        return []
    
    def plan_path_between_markers(self, marker_coords):
        """
        Plan paths between consecutive markers.
        If only 1 marker, create a virtual endpoint.
        
        Args:
            marker_coords: List of marker coordinates (x, y)
            
        Returns:
            List of paths between markers
        """
        paths = []
        
        # FIX: Handle single marker case
        if len(marker_coords) < 1:
            logger.warning("No markers found for path planning")
            return paths
        
        if len(marker_coords) == 1:
            logger.info("Only 1 marker found, creating path to virtual endpoint...")
            marker = marker_coords[0]
            # Create virtual endpoint (200 pixels to the right and down)
            virtual_endpoint = (
                min(marker[0] + 200, self.width - 1),
                min(marker[1] + 200, self.height - 1)
            )
            marker_coords = marker_coords + [virtual_endpoint]
            logger.info(f"Added virtual endpoint at {virtual_endpoint}")
        
        # Plan paths between consecutive markers
        for i in range(len(marker_coords) - 1):
            start = marker_coords[i]
            goal = marker_coords[i + 1]
            
            # Clamp coordinates to image bounds
            start = (max(0, min(start[0], self.height - 1)), 
                    max(0, min(start[1], self.width - 1)))
            goal = (max(0, min(goal[0], self.height - 1)), 
                   max(0, min(goal[1], self.width - 1)))
            
            logger.info(f"Planning path from Marker {i} {start} to Marker {i+1} {goal}...")
            path = self.a_star_search(start, goal)
            
            if path:
                paths.append({
                    "from_marker": i,
                    "to_marker": i + 1,
                    "waypoints": path,
                    "length": len(path)
                })
                logger.info(f"  Path found with {len(path)} waypoints")
            else:
                logger.warning(f"Could not find path from Marker {i} to Marker {i+1}")
        
        self.paths = paths
        return paths
    
    def smooth_path(self, path, iterations=5):
        """
        Smooth path using line simplification.
        
        Args:
            path: List of waypoints
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = path.copy()
        
        for _ in range(iterations):
            new_path = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                prev = np.array(smoothed[i - 1], dtype=np.float32)
                curr = np.array(smoothed[i], dtype=np.float32)
                next_pt = np.array(smoothed[i + 1], dtype=np.float32)
                
                # Check if middle point can be skipped
                line_vec = next_pt - prev
                line_len = np.linalg.norm(line_vec)
                
                if line_len > 0:
                    dist_to_line = abs(np.cross(line_vec, prev - curr)) / line_len
                    
                    if dist_to_line > 2:  # Keep if far from straight line
                        new_path.append(smoothed[i])
            
            new_path.append(smoothed[-1])
            smoothed = new_path
        
        return smoothed
    
    def calculate_path_metrics(self, path):
        """Calculate path metrics (length, average cost, etc)."""
        if not path:
            return None
        
        waypoints = path["waypoints"]
        path_length = 0
        total_cost = 0
        
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i + 1])
            path_length += np.linalg.norm(p2 - p1)
            total_cost += self.costmap[p2[0], p2[1]]
        
        avg_cost = total_cost / len(waypoints) if waypoints else 0
        estimated_time = path_length / 0.5  # Assuming 0.5 m/s robot speed
        
        return {
            "path_length": float(path_length),
            "average_cost": float(avg_cost),
            "total_waypoints": len(waypoints),
            "estimated_traversal_time_sec": float(estimated_time)
        }
    
    def save_paths_to_json(self, output_path):
        """Save planned paths to JSON."""
        paths_data = []
        
        for path in self.paths:
            metrics = self.calculate_path_metrics(path)
            paths_data.append({
                "from_marker": path["from_marker"],
                "to_marker": path["to_marker"],
                "waypoints": [(int(w[0]), int(w[1])) for w in path["waypoints"]],
                "metrics": metrics
            })
        
        with open(output_path, 'w') as f:
            json.dump(paths_data, f, indent=4)
        
        logger.info(f"Paths saved to {output_path}")
    
    def draw_paths_on_image(self, image, marker_coords, virtual_endpoint=None):
        """Draw planned paths on image."""
        result = image.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for path_idx, path in enumerate(self.paths):
            color = colors[path_idx % len(colors)]
            waypoints = path["waypoints"]
            
            # Draw path
            for i in range(len(waypoints) - 1):
                pt1 = tuple([int(waypoints[i][1]), int(waypoints[i][0])])
                pt2 = tuple([int(waypoints[i + 1][1]), int(waypoints[i + 1][0])])
                cv2.line(result, pt1, pt2, color, 2)
            
            # Draw direction arrows
            for i in range(0, len(waypoints) - 1, max(1, len(waypoints) // 10)):
                pt1 = tuple([int(waypoints[i][1]), int(waypoints[i][0])])
                pt2 = tuple([int(waypoints[i + 1][1]), int(waypoints[i + 1][0])])
                cv2.arrowedLine(result, pt1, pt2, color, 2, tipLength=0.3)
        
        # Draw actual markers
        for marker_idx, marker in enumerate(marker_coords):
            pt = tuple([int(marker[1]), int(marker[0])])
            cv2.circle(result, pt, 15, (0, 255, 255), 3)
            cv2.putText(result, f"M{marker_idx}", (pt[0] + 20, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw virtual endpoint if exists
        if virtual_endpoint:
            pt = tuple([int(virtual_endpoint[1]), int(virtual_endpoint[0])])
            cv2.circle(result, pt, 12, (255, 0, 255), 2)
            cv2.putText(result, "Virtual", (pt[0] + 20, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return result


def run_stage3(stitched_image_path, marker_coords_json, output_dir):
    """
    Execute Stage 3: Costmap Generation and Path Planning
    
    Args:
        stitched_image_path: Path to stitched image
        marker_coords_json: Path to marker coordinates JSON
        output_dir: Directory to save results
        
    Returns:
        Path planner object or None if failed
    """
    try:
        logger.info("=" * 60)
        logger.info("STAGE 3: COSTMAP & PATH PLANNING")
        logger.info("=" * 60)
        
        # Load image
        image = cv2.imread(stitched_image_path)
        if image is None:
            logger.error(f"Could not load image: {stitched_image_path}")
            return None
        
        # Generate costmap
        costmap_gen = CostmapGenerator(grid_size=50)
        costmap = costmap_gen.generate_costmap(image)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save costmap
        costmap_color = costmap_gen.save_costmap(str(output_dir / "costmap_visualization.png"))
        
        # Load marker coordinates
        marker_coords = []
        virtual_endpoint = None
        
        try:
            with open(marker_coords_json, 'r') as f:
                marker_data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(marker_data, list):
                # Format: [{"x": ..., "y": ...}, ...]
                marker_coords = [(int(m["x"]), int(m["y"])) for m in marker_data]
            elif isinstance(marker_data, dict) and "markers" in marker_data:
                # Format: {"markers": [{"x_pixel": ..., "y_pixel": ...}, ...]}
                marker_coords = [(int(m["x_pixel"]), int(m["y_pixel"])) 
                                for m in marker_data["markers"]]
            else:
                logger.error("Unknown marker JSON format")
                return None
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading marker coordinates: {e}")
            return None
        
        logger.info(f"Loaded {len(marker_coords)} marker coordinates")
        
        if len(marker_coords) == 0:
            logger.warning("No markers loaded, creating empty path")
            paths_data = []
            with open(output_dir / "planned_paths.json", 'w') as f:
                json.dump(paths_data, f, indent=4)
            return None
        
        # Plan paths
        planner = PathPlanner(costmap, grid_size=50)
        
        # Store original marker count before virtual endpoint is added
        original_marker_count = len(marker_coords)
        
        # FIX: Handle single marker case in plan_path_between_markers
        paths = planner.plan_path_between_markers(marker_coords)
        
        # Check if virtual endpoint was added
        if len(planner.paths) > 0 and len(marker_coords) > original_marker_count:
            virtual_endpoint = marker_coords[-1]
        
        # Save path results
        planner.save_paths_to_json(str(output_dir / "planned_paths.json"))
        
        # Draw paths on image
        paths_image = planner.draw_paths_on_image(image, marker_coords[:original_marker_count], virtual_endpoint)
        cv2.imwrite(str(output_dir / "map_with_planned_paths.png"), paths_image)
        logger.info(f"Path visualization saved to map_with_planned_paths.png")
        
        # Log path metrics
        logger.info("\n" + "=" * 60)
        logger.info("PATH PLANNING RESULTS")
        logger.info("=" * 60)
        
        if planner.paths:
            for path in planner.paths:
                metrics = planner.calculate_path_metrics(path)
                logger.info(f"\nPath {path['from_marker']} → {path['to_marker']}:")
                logger.info(f"  Length: {metrics['path_length']:.2f} pixels")
                logger.info(f"  Average Cost: {metrics['average_cost']:.2f}")
                logger.info(f"  Total Waypoints: {metrics['total_waypoints']}")
                logger.info(f"  Est. Time: {metrics['estimated_traversal_time_sec']:.2f} seconds")
        else:
            logger.warning("No paths could be planned")
        
        logger.info("=" * 60)
        logger.info("✓ Stage 3 completed successfully")
        logger.info("=" * 60)
        
        return planner
        
    except Exception as e:
        logger.error(f"Stage 3 failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Example usage
    stitched_image_path = "./output/stage1_stitched_map.png"
    marker_coords_json = "./output/marker_coordinates.json"
    output_dir = "./output"
    
    planner = run_stage3(stitched_image_path, marker_coords_json, output_dir)