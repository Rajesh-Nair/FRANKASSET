# logger/__init__.py
from .custom_logger import CustomLogger
from pathlib import Path
import os

# Determine the base directory
BASE_DIR = Path(__file__).resolve().parent.parent
project = os.path.basename(BASE_DIR)

# Create a single shared logger instance
GLOBAL_LOGGER = CustomLogger().get_logger(project)
print(f"Logger initialized for project: {project}")
