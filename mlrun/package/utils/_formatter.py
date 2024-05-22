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
import ast
import json
from abc import ABC, abstractmethod
from typing import Any, Union

import yaml

from ._supported_format import SupportedFormat


class _Formatter(ABC):
    """
    An abstract base class for a formatter - a class to format python structures into and from files.
    """

    @classmethod
    @abstractmethod
    def write(cls, obj: Any, file_path: str, **dump_kwargs: dict):
        """
        Write the object to a file. The object must be serializable according to the used format.

        :param obj:         The object to write.
        :param file_path:   The file path to write to.
        :param dump_kwargs: Additional keyword arguments to pass to the dump method of the formatter in use.
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file_path: str) -> Any:
        """
        Read an object from the file given.

        :param file_path: The file to read the object from.

        :return: The read object.
        """
        pass


class _JSONFormatter(_Formatter):
    """
    A static class for managing json files.
    """

    # A set of default configurations to pass to the dump function:
    DEFAULT_DUMP_KWARGS = {"indent": 4}

    @classmethod
    def write(cls, obj: Union[list, dict], file_path: str, **dump_kwargs: dict):
        """
        Write the object to a json file. The object must be serializable according to the json format.

        :param obj:         The object to write.
        :param file_path:   The file path to write to.
        :param dump_kwargs: Additional keyword arguments to pass to the `json.dump` method of the formatter in use.
        """
        dump_kwargs = dump_kwargs or cls.DEFAULT_DUMP_KWARGS
        with open(file_path, "w") as file:
            json.dump(obj, file, **dump_kwargs)

    @classmethod
    def read(cls, file_path: str) -> Union[list, dict]:
        """
        Read an object from the json file given.

        :param file_path: The json file to read the object from.

        :return: The read object.
        """
        with open(file_path) as file:
            obj = json.load(file)
        return obj


class _JSONLFormatter(_Formatter):
    """
    A static class for managing jsonl files.
    """

    @classmethod
    def write(cls, obj: Union[list, dict], file_path: str, **dump_kwargs: dict):
        """
        Write the object to a jsonl file. The object must be serializable according to the json format.

        :param obj:         The object to write.
        :param file_path:   The file path to write to.
        :param dump_kwargs: Additional keyword arguments to pass to the `json.dumps` method of the formatter in use.
        """
        if isinstance(obj, dict):
            obj = [obj]

        with open(file_path, "w") as file:
            for line in obj:
                file.write(json.dumps(obj=line, **dump_kwargs) + "\n")

    @classmethod
    def read(cls, file_path: str) -> Union[list, dict]:
        """
        Read an object from the jsonl file given.

        :param file_path: The jsonl file to read the object from.

        :return: The read object.
        """
        with open(file_path) as file:
            lines = file.readlines()

        obj = []
        for line in lines:
            obj.append(json.loads(s=line))

        return obj[0] if len(obj) == 1 else obj


class _YAMLFormatter(_Formatter):
    """
    A static class for managing yaml files.
    """

    # A set of default configurations to pass to the dump function:
    DEFAULT_DUMP_KWARGS = {"default_flow_style": False, "indent": 4}

    @classmethod
    def write(cls, obj: Union[list, dict], file_path: str, **dump_kwargs: dict):
        """
        Write the object to a yaml file. The object must be serializable according to the yaml format.

        :param obj:         The object to write.
        :param file_path:   The file path to write to.
        :param dump_kwargs: Additional keyword arguments to pass to the `yaml.safe_dump` method of the formatter in use.
        """
        dump_kwargs = dump_kwargs or cls.DEFAULT_DUMP_KWARGS
        with open(file_path, "w") as file:
            yaml.safe_dump(obj, file, **dump_kwargs)

    @classmethod
    def read(cls, file_path: str) -> Union[list, dict]:
        """
        Read an object from the yaml file given.

        :param file_path: The yaml file to read the object from.

        :return: The read object.
        """
        with open(file_path) as file:
            obj = yaml.safe_load(file)
        return obj


class _TXTFormatter(_Formatter):
    """
    A static class for managing txt files.
    """

    @classmethod
    def write(cls, obj: Any, file_path: str, **dump_kwargs: dict):
        """
        Write the object to a text file. The object must be serializable according to python's ast module.

        :param obj:         The object to write.
        :param file_path:   The file path to write to.
        :param dump_kwargs: Ignored.
        """
        with open(file_path, "w") as file:
            file.write(str(obj))

    @classmethod
    def read(cls, file_path: str) -> Any:
        """
        Read an object from the yaml file given.

        :param file_path: The yaml file to read the object from.

        :return: The read object.
        """
        with open(file_path) as file:
            obj = ast.literal_eval(file.read())
        return obj


class StructFileSupportedFormat(SupportedFormat[_Formatter]):
    """
    Library of struct formats (file extensions) supported by some builtin MLRun packagers.
    """

    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    TXT = "txt"

    _FORMAT_HANDLERS_MAP = {
        JSON: _JSONFormatter,
        JSONL: _JSONLFormatter,
        YAML: _YAMLFormatter,
        TXT: _TXTFormatter,
    }
