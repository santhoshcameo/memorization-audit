"""
Configuration Loader
Loads and validates YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy


class ConfigLoader:
    """
    Load and merge configuration files
    
    Supports:
    - Loading from YAML files
    - Merging multiple configs (defaults + specific)
    - Validating required fields
    - Deep dictionary merging
    """
    
    def __init__(self, config_root: Optional[Path] = None):
        """
        Initialize config loader
        
        Args:
            config_root: Root directory for config files (default: ./config)
        """
        if config_root is None:
            # Auto-detect: go up from src/utils to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_root = project_root / "config"
        
        self.config_root = Path(config_root)
        
        if not self.config_root.exists():
            raise ValueError(f"Config root not found: {self.config_root}")
    
    def load(self, config_path: Union[str, Path], 
             merge_defaults: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file (relative to config_root or absolute)
            merge_defaults: Whether to merge with default.yaml
        
        Returns:
            Dictionary with configuration
        """
        # Resolve path
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_root / config_path
        
        # Check exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        # Merge with defaults
        if merge_defaults:
            defaults = self.load_defaults()
            config = self.deep_merge(defaults, config)
        
        return config
    
    def load_defaults(self) -> Dict[str, Any]:
        """Load default configuration"""
        default_path = self.config_root / "default.yaml"
        
        if not default_path.exists():
            return {}
        
        with open(default_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        return defaults if defaults else {}
    
    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset configuration"""
        dataset_path = self.config_root / "datasets" / f"{dataset_name}.yaml"
        return self.load(dataset_path, merge_defaults=False)
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load model configuration"""
        model_path = self.config_root / "models" / f"{model_name}.yaml"
        return self.load(model_path, merge_defaults=False)
    
    def load_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        exp_path = self.config_root / "experiments" / f"{experiment_name}.yaml"
        return self.load(exp_path, merge_defaults=True)
    
    def load_full_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load complete configuration for an experiment
        Merges: defaults + experiment + dataset + models
        
        Args:
            experiment_name: Name of experiment config
        
        Returns:
            Complete merged configuration
        """
        # Load experiment config (already includes defaults)
        config = self.load_experiment(experiment_name)
        
        # Load dataset config
        if 'dataset' in config:
            dataset_name = config['dataset']
            dataset_config = self.load_dataset(dataset_name)
            config['dataset_config'] = dataset_config
        
        # Load model configs
        if 'models' in config:
            model_names = config['models']
            if isinstance(model_names, str):
                model_names = [model_names]
            
            config['model_configs'] = {}
            for model_name in model_names:
                model_config = self.load_model(model_name)
                config['model_configs'][model_name] = model_config
        elif 'model' in config:
            # Single model
            model_name = config['model']
            model_config = self.load_model(model_name)
            config['model_configs'] = {model_name: model_config}
        
        return config
    
    @staticmethod
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries
        override values take precedence over base
        
        Args:
            base: Base dictionary
            override: Override dictionary
        
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = ConfigLoader.deep_merge(result[key], value)
            else:
                # Override value
                result[key] = copy.deepcopy(value)
        
        return result
    
    def validate_config(self, config: Dict, required_keys: list) -> bool:
        """
        Validate that config has required keys
        
        Args:
            config: Configuration dictionary
            required_keys: List of required key paths (e.g., ['training.epochs'])
        
        Returns:
            True if valid
        
        Raises:
            ValueError if missing required keys
        """
        missing = []
        
        for key_path in required_keys:
            keys = key_path.split('.')
            current = config
            
            for key in keys:
                if key not in current:
                    missing.append(key_path)
                    break
                current = current[key]
        
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        
        return True
    
    def save_config(self, config: Dict, output_path: Union[str, Path]):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            output_path: Path to save file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


# Convenience functions
def load_config(config_path: str, config_root: Optional[Path] = None) -> Dict:
    """
    Quick function to load a config file
    
    Args:
        config_path: Path to config file
        config_root: Root config directory
    
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_root)
    return loader.load(config_path)


def load_experiment_config(experiment_name: str, 
                           config_root: Optional[Path] = None) -> Dict:
    """
    Quick function to load complete experiment configuration
    
    Args:
        experiment_name: Name of experiment
        config_root: Root config directory
    
    Returns:
        Complete configuration with all merged configs
    """
    loader = ConfigLoader(config_root)
    return loader.load_full_config(experiment_name)


# Example usage
if __name__ == "__main__":
    # Test loading
    loader = ConfigLoader()
    
    # Load experiment
    config = loader.load_full_config("baseline")
    
    print("Loaded configuration:")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Models: {config['models']}")
    print(f"Dataset config loaded: {'dataset_config' in config}")
    print(f"Model configs loaded: {list(config.get('model_configs', {}).keys())}")