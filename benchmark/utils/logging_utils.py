"""
Logging utilities for the benchmark package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                 console_output: bool = True, progress_bar_compatible: bool = True):
    """
    Setup logging configuration with progress bar compatibility.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        progress_bar_compatible: If True, use stderr for console output to avoid interfering with progress bars
    """
    # Create logger
    logger = logging.getLogger("argument_mining_benchmark")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    if console_output:
        if progress_bar_compatible:
            # Use stderr to avoid interfering with progress bars on stdout
            console_handler = logging.StreamHandler(sys.stderr)
        else:
            console_handler = logging.StreamHandler()
        
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "argument_mining_benchmark") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_initialization(logger: logging.Logger, component: str, status: str, details: str = ""):
    """Log initialization status with consistent formatting."""
    if status.lower() == "success":
        logger.info(f"[OK] {component} initialized successfully{f': {details}' if details else ''}")
    elif status.lower() == "failed":
        logger.error(f"[FAIL] {component} initialization failed{f': {details}' if details else ''}")
    elif status.lower() == "disabled":
        logger.info(f"[DISABLED] {component} disabled{f': {details}' if details else ''}")
    else:
        logger.info(f"{component}: {status}{f' - {details}' if details else ''}")


def log_benchmark_progress(logger: logging.Logger, task: str, implementation: str, 
                          status: str, details: str = ""):
    """Log benchmark progress with consistent formatting."""
    if status.lower() == "started":
        logger.info(f"[RUNNING] {task} with {implementation}...")
    elif status.lower() == "completed":
        logger.info(f"[OK] Completed {task} with {implementation}{f': {details}' if details else ''}")
    elif status.lower() == "failed":
        logger.error(f"[FAIL] Failed {task} with {implementation}{f': {details}' if details else ''}")
    elif status.lower() == "skipped":
        logger.info(f"[SKIP] Skipped {task} with {implementation}{f': {details}' if details else ''}")
    else:
        logger.info(f"{task} with {implementation}: {status}{f' - {details}' if details else ''}")
