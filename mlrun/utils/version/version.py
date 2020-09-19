import json
import typing
import importlib.resources


def _resolve_version_info() -> typing.Dict[str, str]:
    return json.loads(
        importlib.resources.read_text("mlrun.utils.version", "version.json")
    )


version_info: typing.Dict[str, str] = _resolve_version_info()
