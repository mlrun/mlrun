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
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Type, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem

from .utils import TypeHintUtils


# TODO: When 3.7 is no longer supported, add "Packager" as reference type hint to cls (cls: Type["Packager"]) and other.
class _PackagerMeta(ABCMeta):
    """
    Metaclass for `Packager` to override type class methods.
    """

    def __lt__(cls, other) -> bool:
        """
        A less than implementation to compare by priority in order to be able to sort the packagers by it.

        :param other: The compared packager.

        :return: True if priority is lower (means better) and False otherwise.
        """
        return cls.PRIORITY < other.PRIORITY

    def __repr__(cls) -> str:
        """
        Get the string representation of a packager in the following format:
        <packager name>(type=<handled type>, artifact_types=[<all supported artifact types>], priority=<priority>)

        :return: The string representation of e packager.
        """
        # Get the packager info into variables:
        packager_name = cls.__name__
        handled_type = (
            (
                # Types have __name__ attribute but typing's types do not.
                cls.PACKABLE_OBJECT_TYPE.__name__
                if hasattr(cls.PACKABLE_OBJECT_TYPE, "__name__")
                else str(cls.PACKABLE_OBJECT_TYPE)
            )
            if cls.PACKABLE_OBJECT_TYPE is not ...
            else "Any"
        )
        supported_artifact_types = cls.get_supported_artifact_types()

        # Return the string representation in the format noted above:
        return (
            f"{packager_name}(packable_type={handled_type}, artifact_types={supported_artifact_types}, "
            f"priority={cls.PRIORITY})"
        )


class Packager(ABC, metaclass=_PackagerMeta):
    """
    The abstract base class for a packager. A packager is a static class that have two main duties:

    1. Packing - get an object that was returned from a function and log it to MLRun. The user can specify packing
       configurations to the packager using log hints. The packed object can be an artifact or a result.
    2. Unpacking - get a ``mlrun.DataItem`` (an input to a MLRun function) and parse it to the desired hinted type. The
       packager is using the instructions it noted itself when originally packing the object.

    The Packager has one class variable and five class methods that must be implemented:

    * ``PACKABLE_OBJECT_TYPE`` - A class variable to specify the object type this packager handles. Used for the
      ``is_packable`` and ``repr`` methods. An ellipses (`...`) means any type.
    * ``PRIORITY`` - The priority of this packager among the rest of the packagers. Should be an integer between 1-10
      where 1 is the highest priority and 10 is the lowest. If not set, a default priority of 5 is set for MLRun
      builtin packagers and 3 for user custom packagers.
    * ``get_default_packing_artifact_type`` - A class method to get the default artifact type for packing an object
      when it is not provided by the user.
    * ``get_default_unpacking_artifact_type`` - A class method to get the default artifact type for unpacking a data
      item when it is not representing a package, but a simple url or an old / manually logged artifact
    * ``get_supported_artifact_types`` - A class method to get the supported artifact types this packager can pack an
      object as. Used for the ``is_packable`` and `repr` methods.
    * ``pack`` - A class method to pack a returned object using the provided log hint configurations while noting itself
      instructions for how to unpack it once needed (only relevant of packed artifacts as results do not need
      unpacking).
    * ``unpack`` - A class method to unpack a MLRun ``DataItem``, parsing it to its desired hinted type using the
      instructions noted while originally packing it.

    The class methods ``is_packable`` and ``is_unpackable`` are implemented with the following basic logic:

    * ``is_packable`` - a class method to know whether to use this packager to pack an object by its
      type and artifact type, compares the object's type with the ``PACKABLE_OBJECT_TYPE`` and checks the artifact type
      is in the returned supported artifacts list from ``get_supported_artifact_types``.
    * ``is_unpackable`` - a class method to know whether to use this packager to unpack a data item by the user noted
      type hint and optionally stored artifact type in the data item (in case it was packaged before), matches the
      ``PACKABLE_OBJECT_TYPE`` to the type hint given (same logic as IDE matchups, meaning subclasses considered as
      unpackable) and checks if the artifact type is in the returned supported artifacts list from
      ``get_supported_artifact_types``.

    Preferably, each packager should handle a single type of object.

    **Linking Artifacts (extra data)**

    In order to link between packages (using the extra data or metrics spec attributes of an artifact), you should use
    the key as if it exists and as value ellipses (...). The manager will link all packages once it is done packing.

    For example, given extra data keys in the log hint as `extra_data`, setting them to an artifact should be::

        artifact = Artifact(key="my_artifact")
        artifact.spec.extra_data = {key: ... for key in extra_data}

    **Clearing Outputs**

    Some of the packagers may produce files and temporary directories that should be deleted once done with logging the
    artifact. The packager can mark paths of files and directories to delete after logging using the class method
    ``future_clear``.

    For example, in the following packager's ``pack`` method we can write a text file, create an Artifact and then mark
    the text file to be deleted once the artifact is logged::

        with open("./some_file.txt", "w") as file:
            file.write("Pack me")
        artifact = Artifact(key="my_artifact")
        cls.add_future_clearing_path(path="./some_file.txt")
        return artifact, None
    """

    #: The type of object this packager can pack and unpack.
    PACKABLE_OBJECT_TYPE: Type = ...

    #: The priority of this packager in the packagers collection of the manager (lower is better).
    PRIORITY: int = ...

    # List of all paths to be deleted by the manager of this packager post logging the packages:
    _CLEARING_PATH_LIST: List[str] = []

    @classmethod
    @abstractmethod
    def get_default_packing_artifact_type(cls, obj: Any) -> str:
        """
        Get the default artifact type used for packing. The method will be used when an object is sent for packing
        without an artifact type noted by the user.

        :param obj: The about to be packed object.

        :return: The default artifact type.
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_unpacking_artifact_type(cls, data_item: DataItem) -> str:
        """
        Get the default artifact type used for unpacking a data item holding an object of this packager. The method will
        be used when a data item is sent for unpacking without it being a package, but a simple url or an old / manually
        logged artifact.

        :param data_item: The about to be unpacked data item.

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
        cls, obj: Any, artifact_type: str = None, configurations: dict = None
    ) -> Union[Tuple[Artifact, dict], dict]:
        """
        Pack an object as the given artifact type using the provided configurations.

        :param obj:            The object to pack.
        :param artifact_type:  Artifact type to log to MLRun.
        :param configurations: Log hints configurations to pass to the packing method.

        :return: If the packed object is an artifact, a tuple of the packed artifact and unpacking instructions
                 dictionary. If the packed object is a result, a dictionary containing the result key and value.
        """
        pass

    @classmethod
    @abstractmethod
    def unpack(
        cls,
        data_item: DataItem,
        artifact_type: str = None,
        instructions: dict = None,
    ) -> Any:
        """
        Unpack the data item's artifact by the provided type using the given instructions.

        :param data_item:     The data input to unpack.
        :param artifact_type: The artifact type to unpack the data item as.
        :param instructions:  Additional instructions noted in the package to pass to the unpacking method.

        :return: The unpacked data item's object.
        """
        pass

    @classmethod
    def is_packable(cls, obj: Any, artifact_type: str = None) -> bool:
        """
        Check if this packager can pack an object of the provided type as the provided artifact type.

        The default implementation check if the packable object type of this packager is equal to the given object's
        type. If it does match, it will look for the artifact type in the list returned from
        `get_supported_artifact_types`.

        :param obj:           The object to pack.
        :param artifact_type: The artifact type to log the object as.

        :return: True if packable and False otherwise.
        """
        # Get the object's type:
        object_type = type(obj)

        # Validate the object type (ellipses means any type):
        if (
            cls.PACKABLE_OBJECT_TYPE is not ...
            and object_type != cls.PACKABLE_OBJECT_TYPE
        ):
            return False

        # Validate the artifact type (if given):
        if artifact_type and artifact_type not in cls.get_supported_artifact_types():
            return False

        return True

    @classmethod
    def is_unpackable(
        cls, data_item: DataItem, type_hint: Type, artifact_type: str = None
    ) -> bool:
        """
        Check if this packager can unpack an input according to the user given type hint and the provided artifact type.

        The default implementation tries to match the packable object type of this packager to the given type hint, if
        it does match, it will look for the artifact type in the list returned from `get_supported_artifact_types`.

        :param data_item:     The input data item to check if unpackable.
        :param type_hint:     The type hint of the input to unpack.
        :param artifact_type: The artifact type to unpack the object as.

        :return: True if unpackable and False otherwise.
        """
        # Check type (ellipses means any type):
        if cls.PACKABLE_OBJECT_TYPE is not ...:
            if not TypeHintUtils.is_matching(
                object_type=cls.PACKABLE_OBJECT_TYPE,
                type_hint=type_hint,
                reduce_type_hint=False,
            ):
                return False

        # Check the artifact type:
        if artifact_type and artifact_type not in cls.get_supported_artifact_types():
            return False

        # Unpackable:
        return True

    @classmethod
    def add_future_clearing_path(cls, path: Union[str, Path]):
        """
        Mark a path to be cleared by this packager's manager post logging the packaged artifacts.

        :param path: The path to clear.
        """
        cls._CLEARING_PATH_LIST.append(str(path))

    @classmethod
    def get_future_clearing_path_list(cls) -> List[str]:
        """
        Get the packager's future clearing path list.

        :return: The clearing path list.
        """
        return cls._CLEARING_PATH_LIST
