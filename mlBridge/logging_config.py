"""
Centralized logging configuration for ML Bridge project.
Provides consistent logging with timestamp, function name, and log level.
"""

import logging
import inspect
import time
from datetime import datetime
from functools import wraps
import sys
import os

# Python's logging system automatically captures function names when called properly

def setup_logger(name=None, level=logging.INFO, log_file=None):
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (defaults to calling module name)
        level: Logging level (default: INFO)
        log_file: Optional file to log to (in addition to console)
    
    Returns:
        logger: Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ml_bridge')
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter with timestamp, function name, level, and message
    # Python's logging automatically detects function names when called properly
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent messages from being passed to ancestor loggers (avoids duplicates)
    logger.propagate = False
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """
    Get or create a logger for the calling module.
    
    Args:
        name: Logger name (defaults to calling module name)
    
    Returns:
        logger: Logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ml_bridge')
    
    return logging.getLogger(name)

# Convenience function to replace print statements
def log_print(*args, level=logging.INFO, sep=' ', end=''):
    """
    Drop-in replacement for print() that uses logging.
    
    Args:
        *args: Arguments to print/log
        level: Logging level (default: INFO)
        sep: Separator between arguments (default: space)
        end: End character (ignored for logging)
    """
    # Get logger for the calling module
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'ml_bridge')
    logger = get_logger(module_name)
    
    # Convert args to string like print() would
    message = sep.join(str(arg) for arg in args)
    
    # Get the calling function name
    caller_name = frame.f_code.co_name
    
    # Use the standard logging methods which automatically capture function names
    if level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    else:
        logger.log(level, message)

# Initialize default logger for the project
def init_project_logging(log_file=None, level=logging.INFO):
    """
    Initialize logging for the entire ML Bridge project.
    
    Args:
        log_file: Optional log file path
        level: Default logging level
    """
    return setup_logger('ml_bridge', level=level, log_file=log_file)


# ---------------------------------------------------------------------------
# Pipeline-script timing banners
# ---------------------------------------------------------------------------
# Used by acbl_*.py (and any other pipeline entry-point) to print a uniform
# Started/Ended/elapsed banner so consolidated chain logs are easy to scan and
# grep. Plain print()s on purpose: pipeline scripts often run before
# init_project_logging() is configured, and the banners need to land on stdout
# regardless of root-logger state.
#
# Usage at the top of `if __name__ == "__main__":`
#     from mlBridge import print_started, print_ended
#     program_start_time = print_started()
#     ...
#     print_ended(program_start_time)
#
# For scripts that wrap their work in try/except, put print_ended() in the
# finally clause so a crashed run still logs how long it ran before failing.

def print_started(label: str = "Started") -> float:
    """Print 'Started: <timestamp>' and return the start time as a float."""
    t0 = time.time()
    print(f"{label}: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}",
          flush=True)
    return t0


def print_ended(t0: float, label: str = "Ended") -> float:
    """Print 'Ended: <timestamp>' and 'Program elapsed time: ...' then return the elapsed seconds."""
    t1 = time.time()
    elapsed = t1 - t0
    print(f"{label}:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))}",
          flush=True)
    print(
        f"Program elapsed time: {elapsed:.1f}s "
        f"({elapsed/60:.1f}min, {elapsed/3600:.2f}h)",
        flush=True,
    )
    return elapsed
