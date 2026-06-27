"""
File Utilities
Path handling and file operations
"""

from pathlib import Path
from typing import Union, List, Optional
import os
import shutil
import json


def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path to project root
    """
    # Go up from src/utils to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root.resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(filepath: Union[str, Path]) -> Path:
    """
    Ensure parent directory of a file exists
    
    Args:
        filepath: File path
    
    Returns:
        File path as Path object
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def list_files(directory: Union[str, Path], 
               extension: Optional[str] = None,
               recursive: bool = False) -> List[Path]:
    """
    List files in directory
    
    Args:
        directory: Directory to search
        extension: File extension filter (e.g., '.jpg', '.yaml')
        recursive: Search recursively
    
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        pattern = f"**/*{extension}" if extension else "**/*"
        files = [f for f in directory.glob(pattern) if f.is_file()]
    else:
        pattern = f"*{extension}" if extension else "*"
        files = [f for f in directory.glob(pattern) if f.is_file()]
    
    return sorted(files)


def copy_file(src: Union[str, Path], 
              dst: Union[str, Path],
              overwrite: bool = False) -> Path:
    """
    Copy file with safety checks
    
    Args:
        src: Source file
        dst: Destination file
        overwrite: Whether to overwrite existing file
    
    Returns:
        Destination path
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists (use overwrite=True): {dst}")
    
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)
    
    return dst


def move_file(src: Union[str, Path], 
              dst: Union[str, Path],
              overwrite: bool = False) -> Path:
    """
    Move file with safety checks
    
    Args:
        src: Source file
        dst: Destination file
        overwrite: Whether to overwrite existing file
    
    Returns:
        Destination path
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists (use overwrite=True): {dst}")
    
    ensure_parent_dir(dst)
    shutil.move(str(src), str(dst))
    
    return dst


def delete_file(filepath: Union[str, Path], missing_ok: bool = True):
    """
    Delete file safely
    
    Args:
        filepath: File to delete
        missing_ok: Don't raise error if file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        if not missing_ok:
            raise FileNotFoundError(f"File not found: {filepath}")
        return
    
    filepath.unlink()


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Args:
        filepath: File path
    
    Returns:
        File size in bytes
    """
    filepath = Path(filepath)
    return filepath.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., '1.5 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def save_json(data: dict, filepath: Union[str, Path], indent: int = 2):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = ensure_parent_dir(filepath)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Load JSON file to dictionary
    
    Args:
        filepath: JSON file path
    
    Returns:
        Dictionary
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)


class PathManager:
    """
    Centralized path management for the project
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path manager
        
        Args:
            project_root: Project root directory (auto-detected if None)
        """
        if project_root is None:
            self.project_root = get_project_root()
        else:
            self.project_root = Path(project_root)
        
        # Define standard paths
        self.config_root = self.project_root / "config"
        self.data_root = self.project_root / "data"
        self.results_root = self.project_root / "results"
        self.src_root = self.project_root / "src"
    
    def get_config_path(self, *parts: str) -> Path:
        """Get path in config directory"""
        return self.config_root.joinpath(*parts)
    
    def get_data_path(self, *parts: str) -> Path:
        """Get path in data directory"""
        return self.data_root.joinpath(*parts)
    
    def get_results_path(self, *parts: str) -> Path:
        """Get path in results directory"""
        return self.results_root.joinpath(*parts)
    
    def get_checkpoint_path(self, model_name: str, experiment_name: str) -> Path:
        """Get checkpoint path for model"""
        path = self.results_root / "checkpoints" / experiment_name / model_name
        ensure_dir(path)
        return path
    
    def get_scores_path(self, experiment_name: str) -> Path:
        """Get scores path for experiment"""
        path = self.results_root / "scores" / experiment_name
        ensure_dir(path)
        return path
    
    def get_figures_path(self, experiment_name: str) -> Path:
        """Get figures path for experiment"""
        path = self.results_root / "figures" / experiment_name
        ensure_dir(path)
        return path
    
    def get_tables_path(self, experiment_name: str) -> Path:
        """Get tables path for experiment"""
        path = self.results_root / "tables" / experiment_name
        ensure_dir(path)
        return path
    
    def get_logs_path(self, experiment_name: str) -> Path:
        """Get logs path for experiment"""
        path = self.results_root / "logs" / experiment_name
        ensure_dir(path)
        return path
    
    def setup_experiment_dirs(self, experiment_name: str) -> dict:
        """
        Setup all directories for an experiment
        
        Args:
            experiment_name: Name of experiment
        
        Returns:
            Dictionary with all paths
        """
        paths = {
            'checkpoints': self.get_checkpoint_path('', experiment_name),
            'scores': self.get_scores_path(experiment_name),
            'figures': self.get_figures_path(experiment_name),
            'tables': self.get_tables_path(experiment_name),
            'logs': self.get_logs_path(experiment_name),
        }
        
        # Create all directories
        for path in paths.values():
            ensure_dir(path)
        
        return paths


# Global path manager instance
_path_manager = None


def get_path_manager() -> PathManager:
    """
    Get global path manager instance
    
    Returns:
        PathManager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


# Example usage
if __name__ == "__main__":
    # Test path manager
    pm = PathManager()
    
    print("Project root:", pm.project_root)
    print("Config root:", pm.config_root)
    print("Data root:", pm.data_root)
    print("Results root:", pm.results_root)
    
    # Setup experiment directories
    paths = pm.setup_experiment_dirs("test_experiment")
    print("\nExperiment paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # List files
    yaml_files = list_files(pm.config_root, extension='.yaml', recursive=True)
    print(f"\nFound {len(yaml_files)} YAML files in config/")