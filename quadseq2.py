import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from quad import QuadrantGrid
from matplotlib.animation import FuncAnimation

class SequenceGenerator:
    def __init__(self, scaffold_grid: QuadrantGrid):
        """Initialize with a scaffolding grid that defines valid point positions."""
        self.scaffold = scaffold_grid.get_grid()
        # Store valid points as a list of tuples
        self.valid_points = []
        for i in range(self.scaffold.shape[0]):
            for j in range(self.scaffold.shape[1]):
                if self.scaffold[i, j] == 1:
                    self.valid_points.append((i, j))
                    
    def generate_random_sequence(self, n_frames: int, points_per_frame: int) -> List[List[Tuple[int, int]]]:
        """Generate a random sequence where points appear on scaffold positions and disappear next frame."""
        sequence = []
        prev_points = []
        
        for _ in range(n_frames):
            # Get available points (points not used in previous frame)
            available_points = [p for p in self.valid_points if p not in prev_points]
            
            # If we don't have enough available points, use all points
            if len(available_points) < points_per_frame:
                available_points = self.valid_points
                
            # Randomly select points
            indices = np.random.choice(len(available_points), 
                                     size=min(points_per_frame, len(available_points)), 
                                     replace=False)
            new_points = [available_points[i] for i in indices]
            
            sequence.append(new_points)
            prev_points = new_points
            
        return sequence

def animate_sequence(scaffold_grid: QuadrantGrid, sequence: List[List[Tuple[int, int]]], 
                    interval: int = 500, title: str = "Grid Sequence"):
    """Animate a sequence of points appearing on the scaffold grid."""
    fig, ax = plt.subplots(figsize=(12, 6))
    scaffold = scaffold_grid.get_grid()
    
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
        
    ani = animation.FuncAnimation(fig, update, frames=len(sequence),
                                interval=interval, repeat=True)
    plt.close()
    
    return HTML(ani.to_jshtml())