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
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Type, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem


class _PackagerMeta(type):
    """
    Metaclass for `Packager` to override type class methods.
    """

    # TODO: When 3.7 is no longer supported, add "Packager" as reference type hint to cls (cls: "Packager")
    def __repr__(cls) -> str:
        """
        Get the string representation of a packager in the following format:
        <packager name> (type=<handled type>, artifact_types=[<all supported artifact types>])

        :return: The string representation of e packager.
        """
        # Get the packager info into variables:
        packager_name = cls.__name__
        handled_type = (
            cls.PACKABLE_OBJECT_TYPE.__name__
            if cls.PACKABLE_OBJECT_TYPE is not ...
            else "Any"
        )
        supported_artifact_types = cls.get_supported_artifact_types()

        # Return the string representation in the format noted above:
        return f"{packager_name} (type={handled_type}, artifact_types={supported_artifact_types}"


class Packager(ABC, metaclass=_PackagerMeta):
    """
    The abstract base class for a packager. A packager is a static class that have two main duties:

    1. Packing - get an object that was returned from a function and log it to MLRun. The user can specify packing
       configurations to the packager using log hints. The packed object can be an artifact or a result.
    2. Unpacking - get a ``mlrun.DataItem`` (an input to a MLRun function) and parse it to the desired hinted type. The
       packager is using the instructions it notetd itself when orignaly packing the object.

    The Packager has one class variable and 4 class methods that must be implemented:

    * Class variable:
       * ``TYPE`` - The object type this packager handles. Defaulted to any type.
    * Class methods:
       * ``get_default_artifact_type`` - Get the default artifact type when it is not provided by the user.
       * ``pack`` - Pack a returned object using the provieded log hint configurations while noting itself instructions
         for how to unpack it once needed (only relevant of packed artifacts as results do not need unpacking).
       * ``unpack`` - Unpack a MLRun ``DataItem``, parsing it to its desired hinted type using the instructions noted
         while originally packing it.
       * ``is_packable`` - Whether to use this packager to pack / unpack an object by the required artifact type.

    Linking Artifacts (extra data)
    ------------------------------

    In order to link between packages (using the extra data or metrics spec attributes of an artifact), you should use
    the key as if it exists and as value ellipses (...). The manager will link all packages once it is done packing.

    For example, given extra data keys in the log hint as `extra_data`, setting them to an artifact should be::

        artifact = Artifact(key="my_artifact")
        artifact.spec.extra_data = {key: ... for key in extra_data}

    Clearing Outputs
    ----------------

    Some of the packagers may produce files and temporary directories that should be deleted once done with logging the
    artifact. The packager can mark paths of files and directories to delete after logging using the class method
    ``future_clear``.

    For example, in the following packager's ``pack`` method we can write a text file, create an Artifact and then mark
    the text file to be deleted once the artifact is logged::

        with open("./some_file.txt", "w") as file:
            file.write("Pack me")
        artifact = Artifact(key="my_artifact")
        cls.future_clear(path="./some_file.txt")
        return artifact, None
    """

    # The type of object this packager can pack and unpack:
    PACKABLE_OBJECT_TYPE: Type = None

    # List of all paths to be deleted by the manager of this packager post logging the packages:
    _CLEARING_PATH_LIST: List[str] = []

    @classmethod
    @abstractmethod
    def get_default_artifact_type(cls, obj: Any) -> str:
        """
        Get the default artifact type of this packager.

        :param obj: The about to be packed object.

        :return: The default artifact type.
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_artifact_types(cls) -> List[str]:
        """
        Get all the supported artifact types on this packager.

        :return: A list of all the supported artifact types.
        """
        pass

    @classmethod
    @abstractmethod
    def pack(
        cls, obj: Any, artifact_type: Union[str, None], configurations: dict
    ) -> Union[Tuple[Artifact, dict], dict, None]:
        """
        Pack an object as the given artifact type using the provided configurations.

        :param obj:            The object to pack.
        :param artifact_type:  Artifact type to log to MLRun.
        :param configurations: Log hints configurations to pass to the packing method.

        :return: If the packed object is an artifact, a tuple of the packed artifact and unpacking instructions
                 dictionary. If the packed object is a result, a dictionary containing the result key and value. If the
                 packager could not pack the object, None is returned.
        """
        pass

    @classmethod
    @abstractmethod
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
        """
        pass

    @classmethod
    @abstractmethod
    def is_packable(cls, object_type: Type, artifact_type: str = None) -> bool:
        """
        Check if this packager can pack an object of the provided type as the provided artifact type.

        :param object_type:   The object type to pack.
        :param artifact_type: The artifact type to log the object as.

        :return: True if packable and False otherwise.
        """
        pass

    @classmethod
    def future_clear(cls, path: str):
        """
        Mark a path to be cleared by this packager's manager post logging the packaged artifacts.

        :param path: The path to clear.
        """
        cls._CLEARING_PATH_LIST.append(path)

    @classmethod
    def get_clearing_path_list(cls) -> List[str]:
        """
        Get the packager's future clearing path list.

        :return: The clearing path list.
        """
        return cls._CLEARING_PATH_LIST
