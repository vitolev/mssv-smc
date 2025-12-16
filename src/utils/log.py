import logging

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging to file and console"""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"{config.experiment_name}")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(config.log_dir / f"{config.experiment_name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
