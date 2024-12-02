import matplotlib.pyplot as plt
from quad.task import *
from quad.vis import visualize_frame, animate_sequence, setup_visualization

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

config = QuadrantConfig() 
seq_gen = SequenceGenerator(scaffold, config)
sequence = seq_gen.generate_random_sequence(n_frames=30, max_points=100,min_lifetime=4, max_lifetime=5)

# animate_sequence(scaffold, sequence, interval=500, title="Points with Lifetimes")
# fig, ax = setup_visualization(scaffold.height, scaffold.width)
last_sample_frame = -4
point_to_sample = None
task_log = {"frame": [], "frame state": [], "available": [], "choice": []}
for frame, active_points in enumerate(sequence):
    frame_state = seq_gen.get_frame_state(active_points, frame)
    print(f"Frame {frame} AVAILABLE: {frame_state.available_actions}")
    if frame_state.available_actions and frame >= last_sample_frame + 4:
        # RANDOM AGENT - INSERT AGENT POLICY HERE
        point_to_sample = random.choice(frame_state.available_actions)

        # INFORMATION FOR AGENT
        color = seq_gen.sample_point(frame_state, point_to_sample)
        print(f"Frame {frame}: CHOICE {point_to_sample}: {color}")
        last_sample_frame = frame
    #visualize_frame(frame_state, scaffold.height, scaffold.width, frame, fig, ax)
    #plt.pause(0.5)

    # logging
    state_matrix = seq_gen.get_state_matrix(frame_state)
    task_log["frame"].append(frame)
    task_log["frame state"].append(state_matrix)
    task_log["available"].append(frame_state.available_actions)
    task_log["choice"].append(point_to_sample)

#plt.ioff()  
#plt.show()

from agents.observers import CompleteObserver

observer = CompleteObserver(seq_gen)
ratios = observer.process_sequence(sequence)

