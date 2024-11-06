import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Union, NamedTuple, Set
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

@dataclass
class ActivePoint:
    """Class to track an active point and its lifetime."""
    position: Tuple[int, int]
    birth_frame: int
    lifetime: int
    
    def is_alive(self, current_frame: int) -> bool:
        """Check if point is still alive at current frame."""
        return current_frame < self.birth_frame + self.lifetime

@dataclass
class FrameAnalysis:
    """Class to store and analyze points in a frame"""
    active_points: Dict[str, List[Tuple[int, int]]]  # Points active in each quadrant
    occupied_positions: Set[Tuple[int, int]]         # All occupied positions
    points_per_quadrant: Dict[str, int]              # Count of points in each quadrant
    total_points: int                                # Total active points
    available_positions: Dict[str, List[Tuple[int, int]]]  # Available positions in each quadrant

    @classmethod
    def from_points(cls, 
                   active_points: List[ActivePoint], 
                   current_frame: int,
                   quadrant_points: Dict[str, List[Tuple[int, int]]]) -> 'FrameAnalysis':
        """
        Create a FrameAnalysis object from a list of active points.
        
        Args:
            active_points: List of ActivePoint objects
            current_frame: Current frame number
            quadrant_points: Dictionary of valid points per quadrant
        
        Returns:
            FrameAnalysis object containing point states and available positions
        """
        # Initialize collections
        active_by_quadrant = defaultdict(list)
        points_per_quadrant = defaultdict(int)
        occupied_positions = set()
        available_by_quadrant = defaultdict(list)
        
        # First, analyze active points
        for point in active_points:
            if point.is_alive(current_frame):
                pos = point.position
                occupied_positions.add(pos)
                
                # Determine which quadrant this point belongs to
                for quadrant, points in quadrant_points.items():
                    if pos in points:
                        active_by_quadrant[quadrant].append(pos)
                        points_per_quadrant[quadrant] += 1
                        break
        
        # Then, determine available positions per quadrant
        for quadrant, valid_points in quadrant_points.items():
            available_by_quadrant[quadrant] = [
                pos for pos in valid_points 
                if pos not in occupied_positions
            ]
        
        return cls(
            active_points=dict(active_by_quadrant),
            occupied_positions=occupied_positions,
            points_per_quadrant=dict(points_per_quadrant),
            total_points=len(occupied_positions),
            available_positions=dict(available_by_quadrant)
        )

class SequenceGenerator:
    def __init__(self, scaffold_grid: QuadrantGrid):
        """Initialize with a scaffolding grid that defines valid point positions."""
        self.scaffold = scaffold_grid.get_grid()
        
        # Store valid points by quadrant
        height, width = self.scaffold.shape
        mid_h, mid_w = height // 2, width // 2
        
        self.quadrant_points = {
            'top_left': [],
            'top_right': [],
            'bottom_left': [],
            'bottom_right': []
        }
        
        for i in range(height):
            for j in range(width):
                if self.scaffold[i, j] == 1:
                    if i < mid_h:
                        if j < mid_w:
                            self.quadrant_points['top_left'].append((i, j))
                        else:
                            self.quadrant_points['top_right'].append((i, j))
                    else:
                        if j < mid_w:
                            self.quadrant_points['bottom_left'].append((i, j))
                        else:
                            self.quadrant_points['bottom_right'].append((i, j))
    
    def generate_random_sequence(self, 
                               n_frames: int, 
                               max_points: int,
                               min_lifetime: int = 3,
                               max_lifetime: int = 8,
                               appearance_prob: float = 0.3) -> List[List[Tuple[int, int]]]:
        """
        Generate a sequence where points appear randomly and persist for multiple frames.
        
        Args:
            n_frames: Number of frames in sequence
            max_points: Maximum number of points that can exist simultaneously
            min_lifetime: Minimum number of frames a point can exist
            max_lifetime: Maximum number of frames a point can exist
            appearance_prob: Probability of a new point appearing in each quadrant per frame
        
        Returns:
            List of frames, where each frame contains list of active point positions
        """
        active_points = []  # List to track all active points
        sequence = []  # Final sequence of frames
        
        for frame in range(n_frames):
            # Remove dead points
            active_points = [p for p in active_points if p.is_alive(frame)]
            
            # Try to add new points if under max_points
            if len(active_points) < max_points:
                # Check each quadrant
                for quadrant, points in self.quadrant_points.items():
                    if random.random() < appearance_prob:
                        # Get currently occupied positions
                        occupied = {p.position for p in active_points}
                        
                        # Get available positions in this quadrant
                        available = [p for p in points if p not in occupied]
                        
                        if available:
                            # Add a new point
                            new_position = random.choice(available)
                            lifetime = random.randint(min_lifetime, max_lifetime)
                            active_points.append(ActivePoint(new_position, frame, lifetime))
            
            # Add current frame to sequence
            sequence.append([p.position for p in active_points])
        
        return sequence


    def analyze_sequence(self, sequence: List[List[Tuple[int, int]]]) -> List[FrameAnalysis]:
        """
        Analyze an entire sequence frame by frame.
        
        Args:
            sequence: List of frames, where each frame contains point positions
            
        Returns:
            List of FrameAnalysis objects, one per frame
        """
        analyses = []
        active_points = []
        
        for frame, positions in enumerate(sequence):
            
            active_points = [p for p in active_points if p.is_alive(frame)]
            for pos in positions:
                if pos not in {p.position for p in active_points}:
                    
                    lifetime = 1
                    for future_frame in range(frame + 1, len(sequence)):
                        if pos in sequence[future_frame]:
                            lifetime += 1
                        else:
                            break
                    active_points.append(ActivePoint(pos, frame, lifetime))
            
            analysis = FrameAnalysis.from_points(active_points, frame, self.quadrant_points)
            analyses.append(analysis)
            
        return analyses
    
    def print_frame_analysis(self, analysis: FrameAnalysis):
        """Print a human-readable analysis of a frame."""
        print("\nFrame Analysis:")
        print("===============")
        print(f"Total active points: {analysis.total_points}")
        print("\nPoints per quadrant:")
        for quadrant, count in analysis.points_per_quadrant.items():
            print(f"  {quadrant}: {count} points")
            print(f"    Active positions: {analysis.active_points.get(quadrant, [])}")
    
    
    
    
    
    
    
    
    
    
    
