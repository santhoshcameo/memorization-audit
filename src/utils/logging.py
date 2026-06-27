"""
Logging Utilities
Custom logging setup for experiments
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for terminal output
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color
        if record.levelname in self.COLORS:
            record.levelname = (f"{self.COLORS[record.levelname]}"
                              f"{record.levelname:8s}"
                              f"{self.RESET}")
        
        return super().format(record)


def setup_logger(name: str = "medical_memorization",
                level: str = "INFO",
                log_file: Optional[Path] = None,
                log_to_console: bool = True,
                log_to_file: bool = True) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, auto-generated)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format strings
    detailed_format = (
        '%(asctime)s | %(levelname)s | %(name)s | '
        '%(filename)s:%(lineno)d | %(message)s'
    )
    simple_format = '%(asctime)s | %(levelname)s | %(message)s'
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            simple_format,
            datefmt=date_format
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_file is None:
            # Auto-generate log file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = Path(f"logs/experiment_{timestamp}.log")
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Use detailed formatter for file
        file_formatter = logging.Formatter(
            detailed_format,
            datefmt=date_format
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class ExperimentLogger:
    """
    Logger for tracking experiment details
    Logs both to standard logging and structured JSON
    """
    
    def __init__(self, experiment_name: str, 
                 output_dir: Path,
                 config: Optional[dict] = None):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of experiment
            output_dir: Directory to save logs
            config: Experiment configuration
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup standard logger
        log_file = self.output_dir / f"{experiment_name}.log"
        self.logger = setup_logger(
            name=experiment_name,
            log_file=log_file
        )
        
        # Initialize experiment log
        self.experiment_log = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'metrics': {},
            'events': []
        }
        
        self.logger.info(f"="*60)
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"="*60)
    
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.experiment_log['config'] = config
        self.logger.info("Configuration loaded")
    
    def log_event(self, event: str, details: Optional[dict] = None):
        """
        Log an event
        
        Args:
            event: Event description
            details: Additional details
        """
        event_log = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details
        }
        self.experiment_log['events'].append(event_log)
        
        msg = f"Event: {event}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/epoch number
        """
        if name not in self.experiment_log['metrics']:
            self.experiment_log['metrics'][name] = []
        
        metric_log = {
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment_log['metrics'][name].append(metric_log)
        
        msg = f"Metric: {name} = {value:.4f}"
        if step is not None:
            msg += f" (step {step})"
        self.logger.info(msg)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics {name: value}
            step: Step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save(self):
        """Save experiment log to JSON"""
        self.experiment_log['end_time'] = datetime.now().isoformat()
        
        log_file = self.output_dir / f"{self.experiment_name}_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        self.logger.info(f"Experiment log saved: {log_file}")
    
    def finalize(self, status: str = "completed"):
        """
        Finalize experiment
        
        Args:
            status: Experiment status (completed, failed, interrupted)
        """
        self.experiment_log['status'] = status
        self.experiment_log['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.experiment_log['start_time'])
        end = datetime.fromisoformat(self.experiment_log['end_time'])
        duration = (end - start).total_seconds()
        self.experiment_log['duration_seconds'] = duration
        
        self.save()
        
        self.logger.info(f"="*60)
        self.logger.info(f"Experiment {status}")
        self.logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        self.logger.info(f"="*60)


def log_system_info(logger: logging.Logger):
    """
    Log system information
    
    Args:
        logger: Logger instance
    """
    import platform
    import torch
    
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info("="*60)


# Example usage
if __name__ == "__main__":
    # Basic logger
    logger = setup_logger("test", level="INFO")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Experiment logger
    exp_logger = ExperimentLogger(
        experiment_name="test_experiment",
        output_dir=Path("logs"),
        config={'model': 'resnet50', 'epochs': 50}
    )
    
    exp_logger.log_event("training_started")
    exp_logger.log_metric("accuracy", 0.85, step=1)
    exp_logger.log_metrics({'loss': 0.5, 'f1': 0.82}, step=1)
    exp_logger.finalize("completed")
    
    # Log system info
    log_system_info(logger)