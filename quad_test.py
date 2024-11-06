from matplotlib import pyplot as plt
from quad import QuadrantGrid, QuadrantPoints
from quadvis import visualize_grid
from quadseq import SequenceGenerator, animate_sequence_matplotlib

scaffold = QuadrantGrid(10, 20)
points = QuadrantPoints(
    top_left=[(0, 0), (0, 2), (0, 4), (0, 6), (0, 8),
              (1,1), (1,3), (1,5), (1,7),
              (2,0), (2,2), (2,4), (2,6), (2,8)],
    bottom_left=[(7, 0), (7, 2), (7, 4), (7, 6), (7, 8),
              (8,1), (8,3), (8,5), (8,7),
              (9,0), (9,2), (9,4), (9,6), (9,8)],
    top_right=[(0, 11), (0, 13), (0, 15), (0, 17), (0, 19),
              (1,12), (1,14), (1,16), (1,18),
              (2,11), (2,13), (2,15), (2,17), (2,19)],
    bottom_right=[(7, 11), (7, 13), (7, 15), (7, 17), (7, 19),
              (8,12), (8,14), (8,16), (8,18),
              (9,11), (9,13), (9,15), (9,17), (9,19)],
    value=1
)
scaffold.place_points(points)
# visualize_grid(scaffold)
# plt.show()

task = QuadrantGrid(10, 20)
seq_gen = SequenceGenerator(scaffold)
sequence = seq_gen.generate_random_sequence(
    n_frames=50,          # Total number of frames
    max_points=5,         # Maximum simultaneous points
    min_lifetime=3,       # Minimum frames a point exists
    max_lifetime=8,       # Maximum frames a point exists
    appearance_prob=0.3   # Probability of new point per quadrant per frame
)
animate_sequence_matplotlib(scaffold, sequence, interval=1000, title="Points with Lifetimes")