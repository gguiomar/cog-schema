import numpy as np
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass

@dataclass
class QuadrantPoints:
    top_left: List[Tuple[int, int]] = None
    top_right: List[Tuple[int, int]] = None
    bottom_left: List[Tuple[int, int]] = None
    bottom_right: List[Tuple[int, int]] = None
    value: int = 1

class QuadrantGrid:
    def __init__(self, height: int, width: int):
        """Initialize a rectangular grid divided into 4 quadrants.
        
        Args:
            height (int): Total height of the grid (must be even)
            width (int): Total width of the grid (must be even)
        """
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Height and width must be even numbers")
        
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        
        # Calculate quadrant boundaries
        self.mid_h = height // 2
        self.mid_w = width // 2
    
    def get_quadrant_bounds(self, quadrant: str) -> Tuple[slice, slice]:
        """Get the slice bounds for a given quadrant."""
        quadrant_map = {
            'top_left': (slice(0, self.mid_h), slice(0, self.mid_w)),
            'top_right': (slice(0, self.mid_h), slice(self.mid_w, self.width)),
            'bottom_left': (slice(self.mid_h, self.height), slice(0, self.mid_w)),
            'bottom_right': (slice(self.mid_h, self.height), slice(self.mid_w, self.width))
        }
        return quadrant_map[quadrant]
    
    def validate_points(self, points: List[Tuple[int, int]], quadrant: str):
        """Validate that points fall within the specified quadrant."""
        if not points:
            return
            
        h_slice, w_slice = self.get_quadrant_bounds(quadrant)
        valid_h = range(h_slice.start, h_slice.stop)
        valid_w = range(w_slice.start, w_slice.stop)
        
        for h, w in points:
            if h not in valid_h or w not in valid_w:
                raise ValueError(f"Point ({h}, {w}) is outside the {quadrant} quadrant bounds")
    
    def place_points(self, points: Union[QuadrantPoints, Dict[str, List[Tuple[int, int]]], List[Tuple[int, int]]], 
                    quadrant: str = None, value: int = 1):
        """Place points in specified quadrant(s) with given value.
        
        Args:
            points: Can be:
                   - QuadrantPoints object
                   - Dict mapping quadrant names to point lists
                   - List of (h, w) points (requires quadrant parameter)
            quadrant: Required if points is a list, ignored otherwise
            value: Value to place at the points (default 1)
        """
        if isinstance(points, QuadrantPoints):
            for quad in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
                quad_points = getattr(points, quad)
                if quad_points:
                    self.validate_points(quad_points, quad)
                    for h, w in quad_points:
                        self.grid[h, w] = points.value
        
        elif isinstance(points, dict):
            for quad, quad_points in points.items():
                if quad_points:
                    self.validate_points(quad_points, quad)
                    for h, w in quad_points:
                        self.grid[h, w] = value
        
        elif isinstance(points, list) and quadrant:
            self.validate_points(points, quadrant)
            for h, w in points:
                self.grid[h, w] = value
        
        else:
            raise ValueError("Invalid points format or missing quadrant specification")
    
    def clear_quadrant(self, quadrant: str):
        """Clear all points in the specified quadrant."""
        h_slice, w_slice = self.get_quadrant_bounds(quadrant)
        self.grid[h_slice, w_slice] = 0
    
    def clear_all(self):
        """Clear the entire grid."""
        self.grid.fill(0)
    
    def get_grid(self) -> np.ndarray:
        """Return the current grid state."""
        return self.grid.copy()

    def __repr__(self):
        """Return string representation of the grid."""
        return str(self.grid)