import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """
    Configures and returns a logger with rotating file handler and console output.
    """
    logger = logging.getLogger("churn_app")
    
    # prevent adding multiple handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=5_000_000,
        backupCount=5
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
