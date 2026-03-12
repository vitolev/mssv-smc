import logging
import os
from pathlib import Path

def find_project_root(start: Path, marker=".gitignore"):
    for parent in [start] + list(start.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")

ROOT_DIR = find_project_root(Path(__file__).resolve())

def setup_main_logging(project_name: str) -> logging.Logger:
    exp_dir = Path(ROOT_DIR) / "experiments" / project_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_dir / "main.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def setup_chain_logging(project_name: str, chain_id: int) -> logging.Logger:
    exp_dir = Path(ROOT_DIR) / "experiments" / project_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{project_name}_chain_{chain_id}")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_dir / f"chain_{chain_id}.log")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
