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
import inspect
import os
import sys
import tempfile
import warnings
from types import MethodType, ModuleType
from typing import Any, Dict, List, Tuple, Type, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunPackageUnpackingError
from mlrun.utils import logger

from ..constants import ArtifactType
from ..packager import Packager


class _Pickler:
    """
    A static class to pickle objects with multiple modules while capturing the environment of the pickled object. The
    pickler will raise warnings in case the object is un-pickled in a mismatching environment (different modules
    and / or python versions)
    """

    @staticmethod
    def pickle(
        obj: Any, pickle_module_name: str, output_path: str = None
    ) -> Tuple[str, Dict[str, Union[str, None]]]:
        """
        Pickle an object using the given module. The pickled object will be saved to file to the given output path.

        :param obj:                The object to pickle.
        :param pickle_module_name: The pickle module to use. For example: "pickle", "joblib", "cloudpickle".
        :param output_path:        The output path to save the 'pkl' file to. If not provided, the pickle will be saved
                                   to a temporary directory. The user is responsible to clean the temporary directory.

        :return: A tuple of the path of the 'pkl' file and the instructions the pickler noted.
        """
        # Get the modules and versions:
        pickle_module = _Pickler._get_module(source=pickle_module_name)
        pickle_module_version = _Pickler._get_module_version(
            module_name=pickle_module_name
        )
        object_module = _Pickler._get_module(source=obj)
        object_module_name = _Pickler._get_module_name(module=object_module)
        object_module_version = _Pickler._get_module_version(
            module_name=object_module_name
        )
        python_version = _Pickler._get_python_version()

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
        object_module_name: str = None,
        python_version: str = None,
        pickle_module_version: str = None,
        object_module_version: str = None,
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
            current_python_version = _Pickler._get_python_version()
            if python_version != current_python_version:
                logger.warn(
                    f"MLRun is trying to load an object that was pickled on python version "
                    f"'{python_version}' but the current python version is '{current_python_version}'. "
                    f"When using pickle, it is recommended to save and load an object on the same python version to "
                    f"reduce unexpected errors."
                )

        # Get the pickle module:
        pickle_module = _Pickler._get_module(source=pickle_module_name)

        # Check the pickle module against the pickled object (only if the version is given):
        if pickle_module_version is not None:
            current_pickle_module_version = _Pickler._get_module_version(
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
            current_object_module_version = _Pickler._get_module_version(
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

        :param module: A module object to get its name. Only the main module name is returned.

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
            if (
                sys.version_info[1] > 7
            ):  # TODO: Remove once Python 3.7 is not supported.
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


# The default pickle module to use for pickling objects:
_DEFAULT_PICKLE_MODULE = "cloudpickle"


class DefaultPackager(Packager):
    """
    A default packager that handles all types and pack them as pickle files.

    The default packager implements all the required methods and have a default logic that should be satisfying most
    use cases. In order to work with this class, you shouldn't override the ``pack``, ``unpack`` and ``is_packable``
    methods, but follow the guidelines below:

    * ``pack`` is getting the object and sending it to the relevant packing method by the artifact type given (if
      artifact type was not provided, the default one will be used). For example: if the artifact type is "object" then
      the class method ``pack_object`` must be implemented. The signature of each pack class method must be::

          @classmethod
          def pack_x(cls, obj: Any, ...) -> Union[Tuple[Artifact, dict], dict, None]:
              pass

      Where 'x' is the artifact type, 'obj' is the object to pack, ... are additional custom log hint configurations and
      the returning values are the packed artifact and the instructions for unpacking it, or in case of result, the
      dictionary of the result with its key and value. The log hint configurations are sent by the user and shouldn't be
      mandatory, meaning they should have a default value.
    * ``unpack`` is getting a ``DataItem`` and sending it to the relevant unpacking method by the artifact type (if
      artifact type was not provided, the default one will be used). For example: if the artifact type stored within
      the ``DataItem`` is "object" then the class method ``unpack_object`` must be implemented. The signature of each
      unpack class method must be::

          @classmethod
          def unpack_x(cls, data_item: mlrun.DataItem, ...) -> Any:
              pass

      Where 'x' is the artifact type, 'data_item' is the artifact's data item to unpack, ... are the instructions that
      were originally returned from ``pack_x`` (Each instruction must be optional (have a default value) to support
      objects from this type that were not packaged but customly logged) and the returning value is the unpacked
      object.
    * ``is_packable`` is getting the object and the artifact type desired to pack and log it as. So, it is automatically
      looking for all pack class methods implemented to collect the supported artifact types. So, if ``PackagerX`` has
      ``pack_y`` and ``pack_z`` that means the artifact types supported are 'y' and 'z'.

    The ``get_default_artifact_type`` method is implemented to return the clss variable ``DEFAULT_ARTIFACT_TYPE``. You
    may still override the method if the default artifact type you need may change according to the object that's about
    to be packaged.

    Remember (from the ``Packager`` docstring):

    * In order to link between packages (using the extra data or metrics spec attributes of an artifact), you should use
      the key as if it exists and as value ellipses (...). The manager will link all packages once it is done packing.

      For example, given extra data keys in the log hint as `extra_data`, setting them to an artifact should be::

          artifact = Artifact(key="my_artifact")
          artifact.spec.extra_data = {key: ... for key in extra_data}
    * Some packagers may produce files and temporary directories that should be deleted once done with logging the
      artifact. The packager can mark paths of files and directories to delete after logging using the class method
      ``future_clear``.

      For example, in the following packager's ``pack`` method we can write a text file, create an Artifact and then
      mark the text file to be deleted once the artifact is logged::

          with open("./some_file.txt", "w") as file:
              file.write("Pack me")
          artifact = Artifact(key="my_artifact")
          cls.future_clear(path="./some_file.txt")
          return artifact, None
    """

    # The type of object this packager can pack and unpack:
    PACKABLE_OBJECT_TYPE: Type = ...
    # The default artifact type to pack as or unpack from:
    DEFAULT_ARTIFACT_TYPE = ArtifactType.OBJECT

    @classmethod
    def get_default_artifact_type(cls, obj: Any) -> str:
        """
        Get the default artifact type of this packager.

        :param obj: The about to be packed object.

        :return: The default artifact type.
        """
        return cls.DEFAULT_ARTIFACT_TYPE

    @classmethod
    def get_supported_artifact_types(cls) -> List[str]:
        """
        Get all the supported artifact types on this packager.

        :return: A list of all the supported artifact types.
        """
        # We look for pack + unpack method couples so there won't be a scenario where an object can be packed but not
        # unpacked. Result has no unpacking so we add it separately.
        return [
            key[len("pack_") :]
            for key in cls.__dict__
            if key.startswith("pack_")
            and f"unpack_{key[len('pack_'):]}" in cls.__dict__
        ] + ["result"]

    @classmethod
    def pack(
        cls, obj: Any, artifact_type: Union[str, None], configurations: dict
    ) -> Union[Tuple[Artifact, dict], dict, None]:
        """
        Pack an object as the given artifact type using the provided configurations.

        :param obj:            The object to pack.
        :param artifact_type:  Artifact type to log to MLRun.
        :param configurations: Log hints configurations to pass to the packing method.

        :return: If the packed object is an artifact, a tuple of the packed artifact and unpacking instructions
                 dictionary. If the packed object is a result, a dictionary containing the result key and value.
        """
        # Get default artifact type in case it was not provided:
        if artifact_type is None:
            artifact_type = cls.get_default_artifact_type(obj=obj)

        # Get the packing method according to the artifact type:
        pack_method = getattr(cls, f"pack_{artifact_type}")

        # Validate correct configurations were passed:
        if not cls._validate_method_arguments(
            method=pack_method,
            arguments=configurations,
            arguments_type=cls._ArgumentsType.CONFIGURATIONS,
        ):
            return None

        # Call the packing method and return the package:
        return pack_method(obj, **configurations)

    @classmethod
    def unpack(
        cls,
        data_item: DataItem,
        artifact_type: str,
        instructions: dict,
    ) -> Any:
        """
        Unpack the data item's artifact by the provided type using the given instructions.

        :param data_item:     The data input to unpack.
        :param artifact_type: The artifact type to unpack the data item as.
        :param instructions:  Additional instructions to pass to the unpacking method.

        :return: The unpacked data item's object.

        :raise MLRunPackageUnpackingError: In case the packager could not unpack the data item.
        """
        # Get the unpacking method according to the artifact type:
        unpack_method = getattr(cls, f"unpack_{artifact_type}")

        # Validate correct instructions were passed:
        if not cls._validate_method_arguments(
            method=unpack_method,
            arguments=instructions,
            arguments_type=cls._ArgumentsType.INSTRUCTIONS,
        ):
            raise MLRunPackageUnpackingError(
                f"The packager '{cls.__name__}' could not unpack the package due to missing instructions. The artifact "
                f"was probably packed with a different packager. Please read the warnings printed for more details."
            )

        # Call the unpacking method and return the object:
        return unpack_method(data_item, **instructions)

    @classmethod
    def is_packable(cls, object_type: Type, artifact_type: str = None) -> bool:
        """
        Check if this packager can pack an object of the provided type as the provided artifact type.

        :param object_type:   The object type to pack.
        :param artifact_type: The artifact type to log the object as.

        :return: True if packable and False otherwise.
        """
        # Check type (ellipses means any type):
        if cls.PACKABLE_OBJECT_TYPE is not Ellipsis and not issubclass(
            object_type, cls.PACKABLE_OBJECT_TYPE
        ):
            return False

        # Check the artifact type:
        if (
            artifact_type is not None
            and artifact_type not in cls.get_supported_artifact_types()
        ):
            return False

        # Packable:
        return True

    @classmethod
    def pack_object(
        cls,
        obj: Any,
        key: str,
        pickle_module_name: str = _DEFAULT_PICKLE_MODULE,
    ) -> Tuple[Artifact, dict]:
        """
        Pack a python object, pickling it into a pkl file and store it in an artifact.

        :param obj:                The object to pack and log.
        :param key:                The artifact's key.
        :param pickle_module_name: The pickle module name to use for serializing the object.

        :return: The artifacts and it's pickling instructions.
        """
        # Pickle the object to file:
        pickle_path, instructions = _Pickler.pickle(
            obj=obj, pickle_module_name=pickle_module_name
        )

        # Initialize an artifact to the pkl file:
        artifact = Artifact(key=key, src_path=pickle_path)

        # Add the pickle path to the clearing list:
        cls.future_clear(path=pickle_path)

        return artifact, instructions

    @classmethod
    def pack_result(cls, obj: Any, key: str) -> dict:
        """
        Pack an object as a result.

        :param obj: The object to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return {key: obj}

    @classmethod
    def unpack_object(
        cls,
        data_item: DataItem,
        pickle_module_name: str = _DEFAULT_PICKLE_MODULE,
        object_module_name: str = None,
        python_version: str = None,
        pickle_module_version: str = None,
        object_module_version: str = None,
    ) -> Any:
        """
        Unpack the data item's object, unpickle it using the instructions and return.

        Warnings of mismatching python and module versions between the original pickling interpreter and this one may be
        raised.

        :param data_item:             The data item holding the pkl file.
        :param pickle_module_name:    Module to use for unpickling the object.
        :param object_module_name:    The original object's module. Used to verify the current interpreter object module
                                      version match the pickled object version before unpickling the object.
        :param python_version:        The python version in which the original object was pickled. Used to verify the
                                      current interpreter python version match the pickled object version before
                                      unpickling the object.
        :param pickle_module_version: The pickle module version. Used to verify the current interpreter module version
                                      match the one who pickled the object before unpickling it.
        :param object_module_version: The original object's module version to match to the interpreter's module version.

        :return: The un-pickled python object.
        """
        # Get the pkl file to local directory:
        pickle_path = data_item.local()

        # Add the pickle path to the clearing list:
        cls.future_clear(path=pickle_path)

        # Unpickle and return:
        return _Pickler.unpickle(
            pickle_path=pickle_path,
            pickle_module_name=pickle_module_name,
            object_module_name=object_module_name,
            python_version=python_version,
            pickle_module_version=pickle_module_version,
            object_module_version=object_module_version,
        )

    class _ArgumentsType:
        """
        Library class for the arguments type to send for `_validate_method_arguments`. Configurations is the term for
        the kwargs sent to packing methods and instructions is the term for the kwargs sent in unpacking methods.
        """

        INSTRUCTIONS = "instructions"
        CONFIGURATIONS = "configurations"

    @classmethod
    def _validate_method_arguments(
        cls, method: MethodType, arguments: dict, arguments_type: str
    ) -> bool:
        """
        Validate keyword arguments to pass to a method. Used for validating log hint configurations for packing methods
        and instructions for unpacking methods.

        :param method:         The method to validate the arguments for.
        :param arguments:      Keyword arguments to validate.
        :param arguments_type: A string to use for the error message. Should be on of the `_ArgumentType` class
                               variables.

        :return: True if all mandatory arguments are given and False otherwise.
        """
        # Get the possible and mandatory (arguments that has no default value) arguments from the functions:
        possible_arguments = inspect.signature(method).parameters
        mandatory_arguments = [
            name
            for name, parameter in possible_arguments
            if parameter.default is not inspect._empty
        ]

        # Validate there are no missing arguments (only mandatory ones):
        missing_arguments = [
            mandatory_argument
            for mandatory_argument in mandatory_arguments
            if mandatory_argument not in arguments
        ]
        if missing_arguments:
            logger.warn(
                f"Missing {arguments_type} for {cls.__name__}: {', '.join(missing_arguments)}. The packager won't "
                f"handle the object."
            )
            return False

        # Validate all given arguments are correct:
        incorrect_arguments = [
            argument for argument in arguments if argument not in possible_arguments
        ]
        if incorrect_arguments:
            logger.warn(
                f"Unexpected {arguments_type} given for {cls.__name__}: {', '.join(incorrect_arguments)}. "
                f"Possible {arguments_type} are: {', '.join(possible_arguments.keys())}. The packager will try to "
                f"continue by ignoring the incorrect arguments."
            )
        return True
