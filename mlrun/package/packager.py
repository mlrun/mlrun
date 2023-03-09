# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib
import sys
import inspect
from abc import ABC
from enum import Enum
from collections import OrderedDict
from types import MethodType, ModuleType
from typing import Any, Dict, List, Union, Type, Tuple
import os
import tempfile
import warnings

import mlrun.errors
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.execution import MLClientCtx


class PickleModule(Enum):
    """
    An enum to specify all supported pickling modules.
    """

    PICKLE = "pickle"
    PICKLE5 = "pickle5"
    JOBLIB = "joblib"
    CLOUDPICKLE = "cloudpickle"


class _Pickler:
    """
    A static class to pickle objects with multiple modules while capturing the environment of the pickled object. The
    pickler will raise warnings in case the object is un-pickled in a mismatching environment (different modules
    and / or python versions)
    """

    class Labels:
        """
        The labels the pickler is noting when pickling an object. According to these labels, when the pickler will try
        to un-pickle the object, it will know to warn the user.
        """

        PICKLE_MODULE_NAME = "pickle_module_name"
        PICKLE_MODULE_VERSION = "pickle_module_version"
        OBJECT_MODULE_NAME = "object_module_name"
        OBJECT_MODULE_VERSION = "object_module_version"
        PYTHON_VERSION = "python_version"

    @staticmethod
    def pickle(
        obj: Any, pickle_module_name: PickleModule, output_path: str = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Pickle an object using the given module. The pickled object will be saved to file to the given output path.

        :param obj:                The object to pickle.
        :param pickle_module_name: The pickle module to use.
        :param output_path:        The output path to save the 'pkl' file to. If not provided, the pickle will be saved
                                   to a temporary directory. The user is responsible to clean the temporary directory.

        :return: A tuple of the path of the 'pkl' file and a dictionary of labels the pickler noted.
        """
        # Get the modules and versions:
        pickle_module = _Pickler._get_module(source=pickle_module_name.value)
        pickle_module_version = _Pickler._get_module_version(
            module_name=pickle_module_name.value
        )
        object_module = _Pickler._get_module(source=obj)
        object_module_name = _Pickler._get_module_name(module=object_module)
        object_module_version = _Pickler._get_module_version(
            module_name=object_module_name
        )
        python_version = _Pickler._get_python_version()

        # Construct the pickler labels dictionary (versions may not be available):
        labels = {
            _Pickler.Labels.OBJECT_MODULE_NAME: object_module_name,
            _Pickler.Labels.PICKLE_MODULE_NAME: pickle_module_name.value,
            _Pickler.Labels.PYTHON_VERSION: python_version,
        }
        if object_module_version is not None:
            labels[_Pickler.Labels.OBJECT_MODULE_VERSION] = object_module_version
        if pickle_module_version is not None:
            labels[_Pickler.Labels.PICKLE_MODULE_VERSION] = pickle_module_version

        # Generate a temporary output path if not provided:
        if output_path is None:
            output_path = os.path.join(tempfile.mkdtemp(), "obj.pkl")

        # Pickle the object to file:
        with open(output_path, "wb") as pkl_file:
            pickle_module.dump(obj, pkl_file)

        return output_path, labels

    @staticmethod
    def unpickle(
        pickle_path: str,
        labels: Dict[str, str],
    ) -> Any:
        """
        Unpickle an object using the given Pickler labels. The module to use for unpickling the object is read from the
        labels and so are the environment the object was originally pickled in. Warnings may be raised in case any of
        the versions are mismatching.

        :param pickle_path: Path to the 'pkl' file to un-pickle.
        :param labels:      Pickler labels noted during the pickling of the object.

        :return: The un-pickled object.
        """
        # Check the python version against the pickled object:
        python_version = _Pickler._get_python_version()
        if labels[_Pickler.Labels.PYTHON_VERSION] != python_version:
            warnings.warn(
                f"MLRun is trying to load an object that was pickled on python version "
                f"'{labels[_Pickler.Labels.PYTHON_VERSION]}' but the current python version is '{python_version}'. "
                f"When using pickle, it is recommended to save and load an object on the same python version to reduce "
                f"unexpected errors."
            )

        # Get the pickle module:
        pickle_module = _Pickler._get_module(
            source=labels[_Pickler.Labels.PICKLE_MODULE_NAME]
        )

        # Check the pickle module against the pickled object (only if the version is given):
        if _Pickler.Labels.PICKLE_MODULE_VERSION in labels:
            pickle_module_version = _Pickler._get_module_version(
                module_name=labels[_Pickler.Labels.PICKLE_MODULE_NAME]
            )
            if labels[_Pickler.Labels.PICKLE_MODULE_VERSION] != pickle_module_version:
                warnings.warn(
                    f"MLRun is trying to load an object that was pickled using "
                    f"{labels[_Pickler.Labels.PICKLE_MODULE_NAME]} version "
                    f"{labels[_Pickler.Labels.PICKLE_MODULE_VERSION]} but the current module version is "
                    f"'{pickle_module_version}'. When using pickle, it is recommended to save and load an object using "
                    f"the same pickling module version to reduce unexpected errors."
                )

        # Check the object module against the pickled object (only if the version is given):
        if _Pickler.Labels.OBJECT_MODULE_VERSION in labels:
            object_module_version = _Pickler._get_module_version(
                module_name=labels[_Pickler.Labels.OBJECT_MODULE_NAME]
            )
            if labels[_Pickler.Labels.OBJECT_MODULE_VERSION] != object_module_version:
                warnings.warn(
                    f"MLRun is trying to load an object from module {labels[_Pickler.Labels.OBJECT_MODULE_NAME]} "
                    f"version {labels[_Pickler.Labels.OBJECT_MODULE_VERSION]} but the current module version is "
                    f"'{object_module_version}'. When using pickle, it is recommended to save and load an object using "
                    f"the same exact module version to reduce unexpected errors."
                )

        # Load the object from the pickle file:
        with open(pickle_path, "rb") as pickle_file:
            obj = pickle_module.load(pickle_file)

        return obj

    @staticmethod
    def _get_module(source: Union[Any, str]) -> ModuleType:
        """
        Get the module from the provided source. The source can be a python object or a module name (string).

        :param source: The source from which to get the module. If python object, the module of the object is returned.
                       If string - meaning module name, the module with this exact name.

        :return: The imported module object.
        """
        if isinstance(source, str):
            return importlib.import_module(source)
        return inspect.getmodule(source)

    @staticmethod
    def _get_module_name(module: ModuleType) -> str:
        """
        Get a module name.

        :param module: A module object to get it's name. The main module name is returned.

        :return: The module's name.
        """
        return module.__name__.split(".")[0]

    @staticmethod
    def _get_module_version(module_name: str) -> Union[str, None]:
        """
        Get a module's version. Most updated modules have versions but some don't. In case the version could not be
        read, None is returned.

        :param module_name: The module's name to get its version.

        :return: The module's version if found and None otherwise.
        """
        # First we'll try to get the module version from `importlib`:
        try:
            # Since Python 3.8, `version` is part of `importlib.metadata`. Before 3.8, we'll use the module
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

        # Secondly, if importlib could not get the version (most likely 'importlib_metadata' is not installed), we'll
        # try to use `pkg_resources` to get the version (the version will be found only if the package name is equal to
        # the module name. For example, if the module name is 'x' then the way we installed the package must be
        # 'pip install x'):
        import pkg_resources

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

        # The version could not be found.
        return None

    @staticmethod
    def _get_python_version() -> str:
        """
        Get the current running python's version.

        :return: The python version string.
        """
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class Packager(ABC):
    """
    An abstract base class for a packager - a static class that handles an object that was given as an input to a MLRun
    function or that was returned from a function and is needed to be logged to MLRun.

    The Packager has two main class methods and two main properties:

    * ``TYPE`` - The object type this packager handles.
    * ``DEFAULT_ARTIFACT_TYPE`` - The default artifact type to be used when the user didn't specify one.
    * ``pack`` - Pack a returned object, logging it to MLRun while noting itself how to unpack it once needed.
    * ``unpack`` - Unpack a MLRun ``DataItem``, parsing it to its desired hinted type using the instructions noted while
      originally packing it.


    """
    TYPE: Type = ...
    DEFAULT_ARTIFACT_TYPE = "object"

    @classmethod
    def pack(
        cls, obj: Any, artifact_type: Union[str, None], instructions: dict
    ) -> Tuple[Artifact, dict]:
        if artifact_type is None:
            artifact_type = cls.DEFAULT_ARTIFACT_TYPE

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
    ) -> Any:
        unpack_method = getattr(cls, f"unpack_{artifact_type}")
        cls._validate_instructions(method=unpack_method, instructions=instructions)
        return unpack_method(data_item, **instructions)

    @classmethod
    def is_packable(cls, object_type: Type, artifact_type: str = None):
        if cls.TYPE is not ... and not issubclass(object_type, cls.TYPE):
            return False
        if (
            artifact_type is not None
            and artifact_type not in cls._get_supported_artifact_types()
        ):
            return False
        return True

    @classmethod
    def pack_object(
        cls,
        obj: Any,
        key: str,
        pickle_module_name: str = PickleModule.CLOUDPICKLE.value,
    ):
        pkl_file, labels = _Pickler.pickle(
            obj=obj, pickle_module_name=PickleModule(pickle_module_name)
        )
        artifact = Artifact(key=key, src_path=pkl_file)
        artifact.metadata.labels = labels

    @classmethod
    def unpack_object(cls, data_item: DataItem, pickler_labels: Dict[str, str]):
        pickle_path = data_item.local()
        return _Pickler.unpickle(pickle_path=pickle_path, labels=pickler_labels)

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
