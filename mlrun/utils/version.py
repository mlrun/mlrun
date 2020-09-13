import json
import typing
from pathlib import Path


def _resolve_version_info() -> typing.Dict[str, str]:
    project_root_path = Path(__file__).absolute().parent.parent.parent
    version_file_path = project_root_path / "version.json"
    with open(version_file_path) as version_file:
        return json.load(version_file)


version_info = _resolve_version_info()


def get_version_info() -> typing.Dict[str, str]:
    return version_info
