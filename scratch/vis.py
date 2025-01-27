import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, NamedTuple
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quad.task import QuadrantGrid, FrameState, PointColor
from quad.task import ActivePoint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_sequence(scaffold_grid: QuadrantGrid, 
                    sequence: List[List[ActivePoint]], 
                    interval: int = 500, 
                    title: str = "Grid Sequence"):
    """
    Animate a sequence of points appearing on the scaffold grid.
    
    Args:
        scaffold_grid: QuadrantGrid object with the scaffold
        sequence: List of frames, where each frame is a list of ActivePoint objects
        interval: Time between frames in milliseconds
        title: Title for the animation
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    scaffold = scaffold_grid.get_grid()
    
    def update(frame):
        ax.clear()
        
        # Show scaffold in light gray
        plt.imshow(scaffold, cmap='Greys', alpha=0.3)
        
        # Create grid for current points
        current_grid = np.zeros((scaffold_grid.height, scaffold_grid.width, 3))
        
        # Plot points with their appropriate colors
        for point in sequence[frame]:
            if point.is_alive(frame):
                x, y = point.position
                if point.sampled:
                    if point.true_color == PointColor.RED:
                        current_grid[x, y] = [1, 0, 0]  # Red
                    else:
                        current_grid[x, y] = [0, 1, 0]  # Green
                else:
                    current_grid[x, y] = [0.5, 0.5, 0.5]  # Gray for unsampled
        
        plt.imshow(current_grid)
        
        # Add grid lines and styling
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, scaffold.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, scaffold.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f"{title} - Frame {frame+1}/{len(sequence)}")
    
    # Create and display animation
    anim = FuncAnimation(fig, update, frames=len(sequence),
                        interval=interval, repeat=True)
    plt.show()

def setup_visualization(height: int, width: int):
    """Set up the visualization figure and axes once."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    return fig, ax

def visualize_frame(frame_state: FrameState, 
                   height: int,
                   width: int,
                   frame: int,
                   fig: plt.Figure,
                   ax: plt.Axes,
                   title: str = "Frame Visualization"):
    """
    Update visualization for current frame.
    """
    ax.clear()
    
    # Create RGB grid for points
    current_grid = np.zeros((height, width, 3))
    
    # Plot points with their appropriate colors
    for quadrant_points in frame_state.active_points.values():
        for point in quadrant_points:
            x, y = point.position
            if point.sampled:
                if point.true_color == PointColor.RED:
                    current_grid[x, y] = [1, 0, 0]  # Red
                else:
                    current_grid[x, y] = [0, 1, 0]  # Green
            else:
                current_grid[x, y] = [0.5, 0.5, 0.5]  # Gray
    
    ax.imshow(current_grid)
    
    # Add grid lines
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_title(f"{title} - Frame {frame}")
    fig.canvas.draw()
    fig.canvas.flush_events()
