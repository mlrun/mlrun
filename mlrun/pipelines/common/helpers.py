import inspect
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, NoReturn


class FlexibleMapper(MutableMapping):
    _external_data: Dict

    def __init__(self, external_data: Any):
        if isinstance(external_data, dict):
            self._external_data = external_data
        elif hasattr(external_data, "to_dict"):
            self._external_data = external_data.to_dict()

    # TODO: decide if we should kill the dict compatibility layer on get, set and del while using dict syntax
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return self._external_data[key]

    def __setitem__(self, key, value) -> NoReturn:
        try:
            setattr(self, key, value)
        except AttributeError:
            self._external_data[key] = value

    def __delitem__(self, key) -> NoReturn:
        try:
            delattr(self, key)
        except AttributeError:
            del self._external_data[key]

    def __len__(self) -> int:
        # TODO: review the intrinsic responsibilities of __len__ on MutableMapping to ensure full compatibility
        return len(self._external_data) + len(vars(self)) - 1

    def __iter__(self) -> Iterator[str]:
        yield from [
            m[0]
            for m in inspect.getmembers(self)
            if not callable(m[1]) and not m[0].startswith("_")
        ]

    def __bool__(self) -> bool:
        return bool(self._external_data)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self}


project_annotation = "mlrun/project"
run_annotation = "mlrun/pipeline-step-type"
function_annotation = "mlrun/function-uri"
