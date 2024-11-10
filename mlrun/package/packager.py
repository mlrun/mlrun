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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem

from .utils import TypeHintUtils


class Packager(ABC):
    """
    The abstract base class for a packager. Packager has two main duties:

    1. **Packing** - get an object that was returned from a function and log it to MLRun. The user can specify packing
       configurations to the packager using log hints. The packed object can be an artifact or a result.
    2. **Unpacking** - get an :py:meth:`mlrun.DataItem<mlrun.datastore.base.DataItem>` (an input to a MLRun function)
       and parse it to the desired hinted type. The packager uses the instructions it noted itself when originally
       packing the object.

    .. rubric:: Custom Implementation (Inherit Packager)

    The Packager has one class variable and five class methods that must be implemented:

    * :py:meth:`PACKABLE_OBJECT_TYPE<PACKABLE_OBJECT_TYPE>` - A class variable to specify the object type this packager
      handles. Used for the ``is_packable`` and ``repr`` methods. An ellipses (`...`) means any type.
    * :py:meth:`PRIORITY<PRIORITY>` - The priority of this packager among the rest of the packagers. Valid values are
      integers between 1-10 where 1 is the highest priority and 10 is the lowest. If not set, a default priority of 5 is
      set for MLRun builtin packagers and 3 for user custom packagers.
    * :py:meth:`get_default_packing_artifact_type` - A class method to get the default artifact type for packing an
      object when it is not provided by the user.
    * :py:meth:`get_default_unpacking_artifact_type` - A class method to get the default artifact type for unpacking a
      data item when it is not representing a package, but a simple url or an old / manually logged artifact.
    * :py:meth:`get_supported_artifact_types` - A class method to get the supported artifact types this packager can
      pack an object as. Used for the ``is_packable`` and `repr` methods.
    * :py:meth:`pack` - A class method to pack a returned object using the provided log hint configurations while noting
      itself instructions for how to unpack it once needed (only relevant for packed artifacts since results do not need
      unpacking).
    * :py:meth:`unpack` - A class method to unpack an MLRun ``DataItem``, parsing it to its desired hinted type using
      the instructions noted while originally packing it.

    The class methods ``is_packable`` and ``is_unpackable`` are implemented with the following basic logic:

    * :py:meth:`is_packable` - a class method to know whether to use this packager to pack an object by its
      type and artifact type. It compares the object's type with the ``PACKABLE_OBJECT_TYPE`` and checks that the
      artifact type is in the returned supported artifacts list from ``get_supported_artifact_types``.
    * :py:meth:`is_unpackable` - a class method to know whether to use this packager to unpack a data item by the user-
      noted type hint and optionally stored artifact type in the data item (in case it was packaged before). It matches
      the ``PACKABLE_OBJECT_TYPE`` to the type hint given (same logic as IDE matchups, meaning subclasses are
      considered as unpackable) and checks if the artifact type is in the returned supported artifacts list from
      ``get_supported_artifact_types``.

    Preferably, each packager should handle a single type of object.

    .. rubric:: Linking Artifacts (extra data)

    To link between packages (using the extra data or metrics spec attributes of an artifact), use
    the key as if it exists and as value ellipses (...). The manager links all packages once it is done packing.

    For example, given extra data keys in the log hint as `extra_data`, setting them to an artifact would be::

        artifact = Artifact(key="my_artifact")
        artifact.spec.extra_data = {key: ... for key in extra_data}

    .. rubric:: Clearing Outputs

    Some of the packagers may produce files and temporary directories that should be deleted once the artifacts
    are logged. The packager can mark paths of files and directories to delete after logging using the class method
    :py:meth:`add_future_clearing_path`.

    For example, in the following packager's ``pack`` method, you can write a text file, create an Artifact, and
    then mark the text file to be deleted once the artifact is logged::

        with open("./some_file.txt", "w") as file:
            file.write("Pack me")
        artifact = Artifact(key="my_artifact")
        self.add_future_clearing_path(path="./some_file.txt")
        return artifact, None
    """

    #: The type of object this packager can pack and unpack.
    PACKABLE_OBJECT_TYPE: type = ...

    #: The priority of this packager in the packagers collection of the manager (lower is better).
    PRIORITY: int = ...

    def __init__(self):
        # Assign the packager's priority (notice that if it is equal to `...` then it will bbe overriden by the packager
        # manager when collected):
        self._priority = Packager.PRIORITY

        # List of all paths to be deleted by the manager of this packager after logging the packages:
        self._future_clearing_path_list: list[str] = []

    @abstractmethod
    def get_default_packing_artifact_type(self, obj: Any) -> str:
        """
        Get the default artifact type used for packing. The method is used when an object is sent for packing
        without an artifact type noted by the user.

        :param obj: The about to be packed object.

        :return: The default artifact type.
        """
        pass

    @abstractmethod
    def get_default_unpacking_artifact_type(self, data_item: DataItem) -> str:
        """
        Get the default artifact type used for unpacking a data item holding an object of this packager. The method
        is used when a data item is sent for unpacking without it being a package, but is a simple url or an old
        / manually logged artifact.

        :param data_item: The about-to-be unpacked data item.

        :return: The default artifact type.
        """
        pass

    @abstractmethod
    def get_supported_artifact_types(self) -> list[str]:
        """
        Get all the supported artifact types on this packager.

        :return: A list of all the supported artifact types.
        """
        pass

    @abstractmethod
    def pack(
        self,
        obj: Any,
        key: Optional[str] = None,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> Union[tuple[Artifact, dict], dict]:
        """
        Pack an object as the given artifact type using the provided configurations.

        :param obj:            The object to pack.
        :param key:            The key of the artifact.
        :param artifact_type:  Artifact type to log to MLRun.
        :param configurations: Log hints configurations to pass to the packing method.

        :return: If the packed object is an artifact, a tuple of the packed artifact and unpacking instructions
                 dictionary. If the packed object is a result, a dictionary containing the result key and value.
        """
        pass

    @abstractmethod
    def unpack(
        self,
        data_item: DataItem,
        artifact_type: Optional[str] = None,
        instructions: Optional[dict] = None,
    ) -> Any:
        """
        Unpack the data item's artifact by the provided type using the given instructions.

        :param data_item:     The data input to unpack.
        :param artifact_type: The artifact type to unpack the data item as.
        :param instructions:  Additional instructions noted in the package to pass to the unpacking method.

        :return: The unpacked data item's object.
        """
        pass

    def is_packable(
        self,
        obj: Any,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> bool:
        """
        Check if this packager can pack an object of the provided type as the provided artifact type.

        The default implementation checks if the packable object type of this packager is equal to the given object's
        type. If it matches, it looks for the artifact type in the list returned from
        `get_supported_artifact_types`.

        :param obj:            The object to pack.
        :param artifact_type:  The artifact type to log the object as.
        :param configurations: The log hint configurations passed by the user.

        :return: True if packable and False otherwise.
        """
        # Get the object's type:
        object_type = type(obj)

        # Validate the object type (ellipses means any type):
        if (
            self.PACKABLE_OBJECT_TYPE is not ...
            and object_type != self.PACKABLE_OBJECT_TYPE
        ):
            return False

        # Validate the artifact type (if given):
        if artifact_type and artifact_type not in self.get_supported_artifact_types():
            return False

        return True

    def is_unpackable(
        self, data_item: DataItem, type_hint: type, artifact_type: Optional[str] = None
    ) -> bool:
        """
        Check if this packager can unpack an input according to the user-given type hint and the provided artifact type.

        The default implementation tries to match the packable object type of this packager to the given type hint. If
        it matches, it looks for the artifact type in the list returned from `get_supported_artifact_types`.

        :param data_item:     The input data item to check if unpackable.
        :param type_hint:     The type hint of the input to unpack (the object type to be unpacked).
        :param artifact_type: The artifact type to unpack the object as.

        :return: True if unpackable and False otherwise.
        """
        # Check type (ellipses means any type):
        if self.PACKABLE_OBJECT_TYPE is not ...:
            if not TypeHintUtils.is_matching(
                object_type=type_hint,  # The type hint is the expected object type the MLRun function wants.
                type_hint=self.PACKABLE_OBJECT_TYPE,
                reduce_type_hint=False,
            ):
                return False

        # Check the artifact type:
        if artifact_type and artifact_type not in self.get_supported_artifact_types():
            return False

        # Unpackable:
        return True

    def add_future_clearing_path(self, path: Union[str, Path]):
        """
        Mark a path to be cleared by this packager's manager after logging the packaged artifacts.

        :param path: The path to clear post logging the artifacts.
        """
        self._future_clearing_path_list.append(str(path))

    @property
    def priority(self) -> int:
        """
        Get the packager's priority.

        :return: The packager's priority.
        """
        return self._priority

    @priority.setter
    def priority(self, priority: int):
        """
        Set the packager's priority.

        :param priority: The priority to set.
        """
        self._priority = priority

    @property
    def future_clearing_path_list(self) -> list[str]:
        """
        Get the packager's future clearing path list.

        :return: The clearing path list.
        """
        return self._future_clearing_path_list

    def __lt__(self, other: "Packager") -> bool:
        """
        A less than implementation to compare by priority in order to be able to sort the packagers by it.

        :param other: The compared packager.

        :return: True if priority is lower (means better) and False otherwise.
        """
        return self.priority < other.priority

    def __repr__(self) -> str:
        """
        Get the string representation of a packager in the following format:
        <packager name>(type=<handled type>, artifact_types=[<all supported artifact types>], priority=<priority>)

        :return: The string representation of e packager.
        """
        # Get the packager info into variables:
        packager_name = self.__class__.__name__
        handled_type = (
            (
                # Types have __name__ attribute but typing's types do not.
                self.PACKABLE_OBJECT_TYPE.__name__
                if hasattr(self.PACKABLE_OBJECT_TYPE, "__name__")
                else str(self.PACKABLE_OBJECT_TYPE)
            )
            if self.PACKABLE_OBJECT_TYPE is not ...
            else "Any"
        )
        supported_artifact_types = self.get_supported_artifact_types()

        # Return the string representation in the format noted above:
        return (
            f"{packager_name}(packable_type={handled_type}, artifact_types={supported_artifact_types}, "
            f"priority={self.priority})"
        )

    def get_data_item_local_path(
        self, data_item: DataItem, add_to_future_clearing_path: Optional[bool] = None
    ) -> str:
        """
        Get the local path to the item handled by the data item provided. The local path can be the same as the data
        item in case the data item points to a local path, or will be downloaded to a temporary directory and return
        this newly created temporary local path.

        :param data_item:                   The data item to get its item local path.
        :param add_to_future_clearing_path: Whether to add the local path to the future clearing paths list. If None, it
                                            will add the path to the list only if the data item is not of kind 'file',
                                            meaning it represents a local file and hence we don't want to delete it post
                                            running automatically. We wish to delete it only if the local path is
                                            temporary (and that will be in case kind is not 'file', so it is being
                                            downloaded to a temporary directory).

        :return: The data item local path.
        """
        # Get the local path to the item handled by the data item (download it to temporary if not local already):
        local_path = data_item.local()

        # Check if needed to add to the future clear list:
        if add_to_future_clearing_path or (
            add_to_future_clearing_path is None and data_item.kind != "file"
        ):
            self.add_future_clearing_path(path=local_path)

        return local_path
