import os
import importlib
import inspect
import sys

__all__ = []

current_dir = os.path.dirname(__file__)
package_name = __name__

for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        full_module_name = f"{package_name}.{module_name}"
        
        # Import the module
        module = importlib.import_module(full_module_name)
        
        # Add module to globals for access
        globals()[module_name] = module
        
        # Extract classes defined in this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only include classes defined in this module (not imported ones)
            if obj.__module__ == full_module_name:
                globals()[name] = obj
                __all__.append(name)