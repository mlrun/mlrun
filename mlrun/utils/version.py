import json
import typing


def _resolve_version_info() -> typing.Dict[str, str]:
    with open("../../version.json") as version_file:
        return json.load(version_file)


version_info = _resolve_version_info()


def get_version_info() -> typing.Dict[str, str]:
    return version_info
