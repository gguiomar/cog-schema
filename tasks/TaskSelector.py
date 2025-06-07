from enum import Enum
from tasks.BiasDetectionTask import BiasDetectionTask
from tasks.ClassicalConditioningTask import ClassicalConditioningTask
from tasks.PatternDetectionTask import PatternDetectionTask

"""To add a new task, create a new class in the tasks directory and add it to this enum.
   Make sure to implement the get_task method to return an instance of the new task class."""
class TaskSelector(Enum):
    BIAS_DETECTION = ("BiasDetection", BiasDetectionTask)
    CLASSICAL_CONDITIONING = ("ClassicalConditioning", ClassicalConditioningTask)
    PATTERN_DETECTION = ("PatternDetection", PatternDetectionTask)

    def __init__(self, task_name: str, task_class):
        self._task_name = task_name
        self._task_class = task_class

    @classmethod
    def get_list(cls):
        """Return a list of task names."""
        return [task._task_name for task in cls]
    @classmethod
    def from_string(cls, task_name: str):
        """Convert a string to a TaskSelector enum member."""
        for task in cls:
            if task._task_name == task_name:
                return task
        raise ValueError(f"Task '{task_name}' not found in TaskSelector.")
    
    def get_task(self, **kwargs):
        """Return an instance of the task class with optional parameters."""
        return self._task_class(**kwargs)
