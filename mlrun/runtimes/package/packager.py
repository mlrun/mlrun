import importlib
import sys
import inspect
from abc import ABC
from enum import Enum
from types import MethodType
from typing import Any, Dict, List, Union, Type

import mlrun.errors
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem


class PickleModule(Enum):
    PICKLE = "pickle"
    PICKLE5 = "pickle5"
    JOBLIB = "joblib"
    CLOUDPICKLE = "cloudpickle"


class Packager(ABC):

    TYPES: Dict[Type, str] = {Any: "object"}
    MODULE: str = None

    @classmethod
    def pack(cls, obj: Any, artifact_type: Union[str, None], instructions: dict) -> Union[Artifact, dict]:
        if artifact_type is None:
            artifact_type = cls.TYPES.get(type(obj), cls.TYPES[Any])

        pack_method = getattr(cls, f"pack_{artifact_type}")
        cls._validate_instructions(method=pack_method, instructions=instructions)
        return pack_method(obj, **instructions)

    @classmethod
    def unpack(
        cls,
        data_item: DataItem,
        artifact_type: str,
        type_hint: Type,
        type_packaged: Type,
        instructions: dict,
    ) -> object:
        unpack_method = getattr(cls, f"unpack_{artifact_type}")
        cls._validate_instructions(method=unpack_method, instructions=instructions)
        return unpack_method(data_item, **instructions)

    @classmethod
    def is_packable(cls, obj: Any, artifact_type: str):
        if cls.MODULE is not None:
            module_name = inspect.getmodule(obj).__name__
            if cls.MODULE not in [
                module_name.rsplit(".", split_count)[0]
                for split_count in range(module_name.count(".") + 1)
            ]:
                return False
        if Any not in cls.TYPES and type(obj) not in cls.TYPES:
            return False
        if (
            artifact_type is not None
            and artifact_type not in cls._get_supported_artifact_types()
        ):
            return False
        return True

    @classmethod
    def pack_object(
        cls, obj: Any, pickle_module_name: str = PickleModule.CLOUDPICKLE.value
    ):
        # TODO: Pickle the object to 'pkl' file and create an artifact out of it. In addition, put extra labels on it
        #       for object module name, object module version, pickle module name, pickle module version, python version
        object_module_name = inspect.getmodule(obj).__name__.split(".")[0]
        object_module_version = cls._get_module_version(module_name=object_module_name)

    @classmethod
    def _get_supported_artifact_types(cls):
        return [key[len("pack_") :] for key in cls.__dict__ if key.startswith("pack_")]

    @classmethod
    def _validate_instructions(cls, method: MethodType, instructions: dict):
        possible_instructions = inspect.signature(method).parameters
        for instruction_key, instruction_value in instructions.items():
            if instruction_key not in possible_instructions:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Unexpected instruction given: '{instruction_key}'. "
                    f"Possible instructions are: {', '.join(possible_instructions.keys())}"
                )

    @staticmethod
    def _get_module_version(module_name: str) -> Union[str, None]:
        # TODO: Move to a private class `_Pickler`
        # First we'll try to get the module version from `importlib`:
        try:
            # Since Python 3.8 `version` is part of `importlib.metadata`. Before 3.8, we'll use the module
            # `importlib_metadata` to get `version`.
            if sys.version_info[1] > 7:
                from importlib.metadata import version
            else:
                from importlib_metadata import version

            return version(module_name)
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError):
            # User won't necessarily have the `importlib_metadata` module, so we will ignore it by catching
            # `ModuleNotFoundError`. `PackageNotFoundError` is ignored as well as this is raised when `version` could
            # not find the package related to the module.
            pass

        # Secondly, if importlib could not get the version, we'll try to use `pkg_resources` to get the version:
        import pkg_resources
        import warnings

        with warnings.catch_warnings():
            # If a module's package is not found, a `PkgResourcesDeprecationWarning` warning will be raised and then
            # `DistributionNotFound` exception will be raised, so we ignore them both:
            warnings.filterwarnings(
                "ignore", category=pkg_resources.PkgResourcesDeprecationWarning
            )
            try:
                return pkg_resources.get_distribution(module_name).version
            except pkg_resources.DistributionNotFound:
                pass
        return None

# builtins_packagers.py
# pandas_packagers.py
# numpy_packagers.py


class NumberPackager(Packager):
    TYPES = [int, float]


class StringPackager(Packager):
    TYPES = [str]


class PathPackager(StringPackager):
    from pathlib import Path

    TYPES = [Path]


class _Pickler:
    pass
