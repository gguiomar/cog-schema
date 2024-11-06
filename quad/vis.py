import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, NamedTuple
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quad.task import QuadrantGrid

def visualize_grid(grid, title="Task Grid", show_coordinates=True, figsize=(10, 8)):
    """Visualize a QuadrantGrid with quadrant boundaries and optional coordinates.
    
    Args:
        grid: QuadrantGrid object
        title: Title for the plot
        show_coordinates: Whether to show coordinate numbers
        figsize: Size of the figure (width, height)
    """
    # Get the grid data
    data = grid.get_grid()
    height, width = data.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the grid
    im = ax.imshow(data, cmap='binary')
    
    # Add quadrant lines
    mid_h, mid_w = height//2, width//2
    ax.axhline(y=mid_h - 0.5, color='red', linestyle='-', alpha=0.5)
    ax.axvline(x=mid_w - 0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Show coordinate numbers if requested
    if show_coordinates:
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Highlight points with values
    for i in range(height):
        for j in range(width):
            if data[i, j] != 0:
                # Add a small black border around the point
                rect = Rectangle((j-0.5, i-0.5), 1, 1, fill=False, color='black', linewidth=1.5)
                ax.add_patch(rect)
                # Add the value in the center of the cell
                ax.text(j, i, str(data[i, j]), ha='center', va='center', 
                       color='black' if data[i, j] == 0 else 'white')
    
    # Set title and layout
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax

def animate_grid_construction(grid, points_sequence, interval=1000):
    """Animate the construction of a grid point by point.
    
    Args:
        grid: QuadrantGrid object
        points_sequence: List of (points, quadrant, value) tuples to animate
        interval: Time between frames in milliseconds
    """
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(10, 8))
    grid.clear_all()
    frames = []
    
    def update(frame):
        points, quadrant, value = points_sequence[frame]
        grid.place_points(points, quadrant=quadrant, value=value)
        ax.clear()
        visualize_grid(grid, show_coordinates=True)
        return ax,
    
    ani = animation.FuncAnimation(fig, update, frames=len(points_sequence),
                                interval=interval, blit=True)
    plt.close()
    return ani


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