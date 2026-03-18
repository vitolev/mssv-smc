from pathlib import Path

# Setting up ROOT_DIR
def find_project_root(start: Path, marker=".gitignore"):
    for parent in [start] + list(start.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")

ROOT_DIR = find_project_root(Path(__file__).resolve())