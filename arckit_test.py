from arckit import load_data
import arckit.vis as vis

train_tasks, eval_tasks = load_data(version="latest")
task = train_tasks[1]  # or any other index
task_drawing = vis.draw_task(task, width=20, height=10)
vis.output_drawing(task_drawing, "task_visualization.png")


## defining the visual sampling task

