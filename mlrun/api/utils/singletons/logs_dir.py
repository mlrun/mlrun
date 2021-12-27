from pathlib import Path

from mlrun.config import config

# TODO: something nicer
logs_dir: Path = None


def get_logs_dir() -> Path:
    global logs_dir
    return logs_dir


def initialize_logs_dir():
    global logs_dir
    logs_dir = Path(config.httpdb.logs_path)
