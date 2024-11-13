import random
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Union, NamedTuple, Set
from dataclasses import dataclass


class PointColor(Enum):
    UNKNOWN = 0
    RED = 1
    GREEN = 2

@dataclass
class QuadrantPoints:
    top_left: List[Tuple[int, int]] = None
    top_right: List[Tuple[int, int]] = None
    bottom_left: List[Tuple[int, int]] = None
    bottom_right: List[Tuple[int, int]] = None
    value: int = 1

@dataclass
class ActivePoint:
    """Enhanced class to track an active point with color identity"""
    position: Tuple[int, int]
    birth_frame: int
    lifetime: int
    true_color: PointColor  # Actual color (hidden from agent)
    visible_color: PointColor = PointColor.UNKNOWN  # Color visible to agent
    sampled: bool = False
    
    def is_alive(self, current_frame: int) -> bool:
        return current_frame < self.birth_frame + self.lifetime
    
    def sample(self) -> PointColor:
        """Sample this point to reveal its true color"""
        self.sampled = True
        self.visible_color = self.true_color
        return self.true_color

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
    
    def get_grid(self) -> np.ndarray:
        """Return a copy of the current grid."""
        return self.grid.copy()
    
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
        """Place points in specified quadrant(s) with given value."""
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


@dataclass
class QuadrantConfig:
    """Configuration for color probabilities in each quadrant"""
    quadrant_probs: Dict[str, Tuple[float, float, int]] = field(default_factory=lambda: {
        'top_left': (0.9, 0.1, 0),      # [90,10,0]
        'top_right': (0.5, 0.5, 1),     # [50,50,1]
        'bottom_left': (0.5, 0.5, 2),   # [50,50,2]
        'bottom_right': (0.5, 0.5, 3)    # [50,50,3]
    })

@dataclass
class FrameState:
    """State of the frame including sampling information"""
    active_points: Dict[str, List[ActivePoint]]  # Points active in each quadrant
    occupied_positions: Set[Tuple[int, int]]     # All occupied positions
    points_per_quadrant: Dict[str, int]          # Count of points per quadrant
    total_points: int                            # Total active points
    available_positions: Dict[str, List[Tuple[int, int]]]  # Available positions
    available_actions: List[Tuple[int, int]]     # Points that can be sampled

class SequenceGenerator:
    def __init__(self, scaffold_grid: QuadrantGrid, config: QuadrantConfig = None):
        """Initialize with scaffold grid and color configuration"""
        self.scaffold = scaffold_grid.get_grid()
        self.config = config or QuadrantConfig()
        self.height, self.width = self.scaffold.shape
        mid_h, mid_w = self.height // 2, self.width // 2
        
        # Initialize quadrant points
        self.quadrant_points = self._init_quadrant_points(mid_h, mid_w)
        
        # Track sampled points for visualization
        self.sampled_points: Dict[Tuple[int, int], PointColor] = {}
    
    def _init_quadrant_points(self, mid_h: int, mid_w: int) -> Dict[str, List[Tuple[int, int]]]:
        quadrant_points = {
            'top_left': [], 'top_right': [], 
            'bottom_left': [], 'bottom_right': []
        }
        
        for i in range(self.height):
            for j in range(self.width):
                if self.scaffold[i, j] == 1:
                    if i < mid_h:
                        if j < mid_w:
                            quadrant_points['top_left'].append((i, j))
                        else:
                            quadrant_points['top_right'].append((i, j))
                    else:
                        if j < mid_w:
                            quadrant_points['bottom_left'].append((i, j))
                        else:
                            quadrant_points['bottom_right'].append((i, j))
        
        return quadrant_points
    
    def _assign_point_color(self, position: Tuple[int, int]) -> PointColor:
        """Assign a color to a point based on its quadrant's probabilities"""

        i, j = position
        mid_h, mid_w = self.height // 2, self.width // 2
        
        if i < mid_h:
            quadrant = 'top_left' if j < mid_w else 'top_right'
        else:
            quadrant = 'bottom_left' if j < mid_w else 'bottom_right'

        red_prob, _, _ = self.config.quadrant_probs[quadrant]
        return PointColor.RED if random.random() < red_prob else PointColor.GREEN
    
    def generate_random_sequence(self, 
                               n_frames: int, 
                               max_points: int,
                               min_lifetime: int = 3,
                               max_lifetime: int = 8,
                               appearance_prob: float = 0.3) -> List[List[ActivePoint]]:
        """Generate sequence with colored points"""
        active_points = []
        sequence = []
        
        for frame in range(n_frames):
            # Remove dead points
            active_points = [p for p in active_points if p.is_alive(frame)]
            
            # Try to add new points
            if len(active_points) < max_points:
                for quadrant, points in self.quadrant_points.items():
                    if random.random() < appearance_prob:
                        occupied = {p.position for p in active_points}
                        available = [p for p in points if p not in occupied]
                        
                        if available:
                            new_position = random.choice(available)
                            lifetime = random.randint(min_lifetime, max_lifetime)
                            true_color = self._assign_point_color(new_position)
                            
                            active_points.append(ActivePoint(
                                position=new_position,
                                birth_frame=frame,
                                lifetime=lifetime,
                                true_color=true_color
                            ))
            
            sequence.append(active_points.copy())
        
        return sequence
    
    def sample_point(self, 
                    frame_state: FrameState, 
                    position: Tuple[int, int]) -> Optional[PointColor]:
        """Sample a point to reveal its color"""
        # Find the point in active points
        for points in frame_state.active_points.values():
            for point in points:
                if point.position == position and not point.sampled:
                    color = point.sample()
                    self.sampled_points[position] = color
                    return color
        return None
    
    def get_frame_state(self, 
                       active_points: List[ActivePoint], 
                       frame: int) -> FrameState:
        """Get the current frame state including available sampling actions"""
        # Initialize collections
        points_by_quadrant = {q: [] for q in self.quadrant_points.keys()}
        occupied = set()
        sampeable_points = []
        
        # Process active points
        for point in active_points:
            if point.is_alive(frame):
                pos = point.position
                occupied.add(pos)
                
                # Add to quadrant
                for quadrant, valid_points in self.quadrant_points.items():
                    if pos in valid_points:
                        points_by_quadrant[quadrant].append(point)
                        break
                
                # Add to sampeable points if not already sampled
                if not point.sampled:
                    sampeable_points.append(pos)
        
        # Calculate available positions
        available = {
            quadrant: [pos for pos in points if pos not in occupied]
            for quadrant, points in self.quadrant_points.items()
        }
        
        return FrameState(
            active_points=points_by_quadrant,
            occupied_positions=occupied,
            points_per_quadrant={q: len(p) for q, p in points_by_quadrant.items()},
            total_points=len(occupied),
            available_positions=available,
            available_actions=sampeable_points
        )
    
    def get_state_matrix(self, frame_state: FrameState) -> np.ndarray:
        """
        Convert a FrameState into a 2D matrix representation.
        
        Returns:
            numpy.ndarray: A height x width matrix where each cell contains:
                0: No point
                1: Unsampled active point
                2: Sampled red point
                3: Sampled green point
        """
        state_matrix = np.zeros((self.height, self.width), dtype=int)
        
        for quadrant_points in frame_state.active_points.values():
            for point in quadrant_points:
                x, y = point.position
                if point.sampled:
                    state_matrix[x, y] = 2 if point.true_color == PointColor.RED else 3
                else:
                    state_matrix[x, y] = 1
                    
        return state_matrix

