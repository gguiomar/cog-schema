import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

from quad.task import QuadrantGrid

@dataclass
class ActivePoint:
    """Class to track an active point and its lifetime."""
    position: Tuple[int, int]
    birth_frame: int
    lifetime: int
    
    def is_alive(self, current_frame: int) -> bool:
        """Check if point is still alive at current frame."""
        return current_frame < self.birth_frame + self.lifetime

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

def visualize_frame_details(sequence: List[List[Tuple[int, int]]], frame_idx: int):
    """Helper function to visualize details about a specific frame."""
    print(f"Frame {frame_idx}:")
    print(f"Number of active points: {len(sequence[frame_idx])}")
    print("Point positions:", sequence[frame_idx])


def animate_sequence_matplotlib(scaffold_grid: QuadrantGrid, 
                              sequence: List[List[Tuple[int, int]]], 
                              interval: int = 500, 
                              title: str = "Grid Sequence",
                              save_path: str = None):
    """
    Animate a sequence of points appearing on the scaffold grid using matplotlib.
    
    Args:
        scaffold_grid: QuadrantGrid object with the scaffold
        sequence: List of frames, where each frame is a list of (x, y) coordinates
        interval: Time between frames in milliseconds
        title: Title for the animation
        save_path: If provided, save the animation to this path (must end in .gif)
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    scaffold = scaffold_grid.get_grid()
    
    def init():
        # Initial frame setup
        ax.clear()
        plt.imshow(scaffold, cmap='Greys', alpha=0.3)
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, scaffold.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, scaffold.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        return []
    
    def update(frame):
        ax.clear()
        
        # Show scaffold in light gray
        plt.imshow(scaffold, cmap='Greys', alpha=0.3)
        
        # Show current points in black
        current_grid = np.zeros_like(scaffold)
        for x, y in sequence[frame]:
            current_grid[x, y] = 1
        plt.imshow(current_grid, cmap='binary')
        
        # Add grid lines and styling
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, scaffold.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, scaffold.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f"{title} - Frame {frame+1}/{len(sequence)}")
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(sequence),
                        init_func=init, interval=interval, 
                        repeat=True, blit=False)
    
    # Save animation if path is provided
    if save_path is not None:
        if not save_path.endswith('.gif'):
            save_path += '.gif'
        anim.save(save_path, writer='pillow')
    
    plt.show()
    
    return anim