# Copyright 2023 Iguazio
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
import importlib.metadata as importlib_metadata
import os
import sys
import tempfile
import warnings
from types import ModuleType
from typing import Any, Optional, Union

from mlrun.errors import MLRunInvalidArgumentError
from mlrun.utils import logger


class Pickler:
    """
    A static class to pickle objects with multiple modules while capturing the environment of the pickled object. The
    pickler will raise warnings in case the object is un-pickled in a mismatching environment (different modules
    and / or python versions)
    """

    @staticmethod
    def pickle(
        obj: Any, pickle_module_name: str, output_path: Optional[str] = None
    ) -> tuple[str, dict[str, Union[str, None]]]:
        """
        Pickle an object using the given module. The pickled object will be saved to file to the given output path.

        :param obj:                The object to pickle.
        :param pickle_module_name: The pickle module to use. For example: "pickle", "joblib", "cloudpickle".
        :param output_path:        The output path to save the 'pkl' file to. If not provided, the pickle will be saved
                                   to a temporary directory. The user is responsible to clean the temporary directory.

        :return: A tuple of the path of the 'pkl' file and the instructions the pickler noted.
        """
        # Get the pickle module:
        pickle_module = importlib.import_module(pickle_module_name)
        Pickler._validate_pickle_module(pickle_module=pickle_module)
        pickle_module_version = Pickler._get_module_version(
            module_name=pickle_module_name
        )

        # Get the object's module (module name can be extracted usually from the object's class):
        object_module_name = (
            obj.__module__.split(".")[0]
            if hasattr(obj, "__module__")
            else type(obj).__module__.split(".")[0]
        )
        object_module_version = Pickler._get_module_version(
            module_name=object_module_name
        )

        # Get the python version:
        python_version = Pickler._get_python_version()

        # Construct the pickler labels dictionary (versions may not be available):
        instructions = {
            "object_module_name": object_module_name,
            "pickle_module_name": pickle_module_name,
            "python_version": python_version,
        }
        if object_module_version is not None:
            instructions["object_module_version"] = object_module_version
        if pickle_module_version is not None:
            instructions["pickle_module_version"] = pickle_module_version

        # Generate a temporary output path if not provided:
        if output_path is None:
            output_path = os.path.join(tempfile.mkdtemp(), "obj.pkl")

        # Pickle the object to file:
        with open(output_path, "wb") as pkl_file:
            pickle_module.dump(obj, pkl_file)

        return output_path, instructions

    @staticmethod
    def unpickle(
        pickle_path: str,
        pickle_module_name: str,
        object_module_name: Optional[str] = None,
        python_version: Optional[str] = None,
        pickle_module_version: Optional[str] = None,
        object_module_version: Optional[str] = None,
    ) -> Any:
        """
        Unpickle an object using the given instructions. Warnings may be raised in case any of the versions are
        mismatching (only if provided - not None).

        :param pickle_path:           Path to the 'pkl' file to un-pickle.
        :param pickle_module_name:    Module to use for unpickling the object.
        :param object_module_name:    The original object's module. Used to verify the current interpreter object module
                                      version match the pickled object version before unpickling the object.
        :param python_version:        The python version in which the original object was pickled. Used to verify the
                                      current interpreter python version match the pickled object version before
                                      unpickling the object.
        :param pickle_module_version: The pickle module version. Used to verify the current interpreter module version
                                      match the one who pickled the object before unpickling it.
        :param object_module_version: The original object's module version to match to the interpreter's module version.

        :return: The un-pickled object.
        """
        # Check the python version against the pickled object:
        if python_version is not None:
            current_python_version = Pickler._get_python_version()
            if python_version != current_python_version:
                logger.warn(
                    f"MLRun is trying to load an object that was pickled on python version "
                    f"'{python_version}' but the current python version is '{current_python_version}'. "
                    f"When using pickle, it is recommended to save and load an object on the same python version to "
                    f"reduce unexpected errors."
                )

        # Get the pickle module:
        pickle_module = importlib.import_module(pickle_module_name)
        Pickler._validate_pickle_module(pickle_module=pickle_module)

        # Check the pickle module against the pickled object (only if the version is given):
        if pickle_module_version is not None:
            current_pickle_module_version = Pickler._get_module_version(
                module_name=pickle_module_name
            )
            if pickle_module_version != current_pickle_module_version:
                logger.warn(
                    f"MLRun is trying to load an object that was pickled using "
                    f"{pickle_module_name} version {pickle_module_version} but the current module version is "
                    f"'{current_pickle_module_version}'. "
                    f"When using pickle, it is recommended to save and load an "
                    f"object using the same pickling module version to reduce unexpected errors."
                )

        # Check the object module against the pickled object (only if the version is given):
        if object_module_version is not None and object_module_name is not None:
            current_object_module_version = Pickler._get_module_version(
                module_name=object_module_name
            )
            if object_module_version != current_object_module_version:
                logger.warn(
                    f"MLRun is trying to load an object from module {object_module_name} version "
                    f"{object_module_version} but the current module version is '{current_object_module_version}'. "
                    f"When using pickle, it is recommended to save and load an object using "
                    f"the same exact module version to reduce unexpected errors."
                )

        # Load the object from the pickle file:
        with open(pickle_path, "rb") as pickle_file:
            obj = pickle_module.load(pickle_file)

        return obj

    @staticmethod
    def _validate_pickle_module(pickle_module: ModuleType):
        """
        Validate the pickle module to use have a `dump` and `load` functions so the Pickler can use it.

        :param pickle_module: The pickle module tot validate.

        :raise MLRunInvalidArgumentError: If the pickle module is not valid.
        """
        for function_name in ["dump", "load"]:
            if not hasattr(pickle_module, function_name):
                raise MLRunInvalidArgumentError(
                    f"A pickle module is expected to have a `{function_name}` function but the provided module "
                    f"{pickle_module.__name__} does not have it."
                )

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
            return importlib_metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            # `PackageNotFoundError` is ignored this is raised when `version` could not find the package related to the
            # module.
            pass

        # Secondly, if importlib could not get the version, we'll try to use `pkg_resources` to get the version (the
        # version will be found only if the package name is equal to the module name. For example, if the module name is
        # 'x' then the way we installed the package must be 'pip install x'):
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
