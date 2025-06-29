"""
Utility functions for the EDML Project
"""

import os
from pathlib import Path


def get_project_root():
    """
    Find the project root directory by looking for specific markers.
    Returns the path to the project root directory.
    """
    current_path = Path(__file__).resolve()
    
    # Look for project root indicators (datasets folder or .git folder)
    for parent in current_path.parents:
        if (parent / "datasets").exists() or (parent / ".git").exists():
            return str(parent)
    
    # Fallback to current file's directory
    return str(current_path.parent)