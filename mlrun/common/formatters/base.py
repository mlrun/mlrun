# Copyright 2024 Iguazio
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

import typing

import mlrun.errors


class ObjectFormat:
    """
    MLRun object formatter. Any class that inherits from this class should implement the `format_method` method
    to specify the formatting method for each format.
    A `filter_obj_method` utility method is provided to filter the object based on a list of keys.
    """

    full = "full"

    @staticmethod
    def format_method(format_: str) -> typing.Optional[typing.Callable]:
        """
        Get the formatting method for the provided format.
        A `None` value signifies a pass-through formatting method (no formatting).
        :param format_: The format as a string representation.
        :return: The formatting method.
        """
        return {
            ObjectFormat.full: None,
        }[format_]

    @classmethod
    def format_obj(
        cls,
        obj: typing.Any,
        format_: str,
        exclude_formats: typing.Optional[list[str]] = None,
    ) -> typing.Any:
        """
        Format the provided object based on the provided format.
        :param obj: The object to format.
        :param format_: The format as a string representation.
        :param exclude_formats: A list of formats to exclude from the formatting process. If the provided format is in
                                this list, an invalid format exception will be raised.
        """
        exclude_formats = exclude_formats or []
        format_ = format_ or cls.full
        invalid_format_exc = mlrun.errors.MLRunBadRequestError(
            f"Provided format is not supported. format={format_}"
        )

        if format_ in exclude_formats:
            raise invalid_format_exc

        try:
            format_method = cls.format_method(format_)
        except KeyError:
            raise invalid_format_exc

        if not format_method:
            return obj

        return format_method(obj)

    @staticmethod
    def filter_obj_method(_filter: list[str]) -> typing.Callable:
        """
        Returns a method that filters the object based on the provided list of keys.
        The keys should be in a dot-separated format, denoting the path within the dictionary to the desired key.
        The object maintains its structure, with the filtered keys and their values, while all other keys are removed.
        :param _filter: The list of keys to filter by.
                        Example:
                        [
                            "kind",
                            "metadata.name",
                            "spec.something.else",
                        ]

        :return: The filtering method.
        """

        def _filter_method(obj: dict) -> dict:
            formatted_obj = {}
            for key in _filter:
                key_list = key.split(".")
                obj_recursive_iterator = obj
                formatted_obj_recursive_iterator = formatted_obj
                for idx, key in enumerate(key_list):
                    if key not in obj_recursive_iterator:
                        break
                    value = (
                        {} if idx < len(key_list) - 1 else obj_recursive_iterator[key]
                    )
                    formatted_obj_recursive_iterator.setdefault(key, value)

                    obj_recursive_iterator = obj_recursive_iterator[key]
                    formatted_obj_recursive_iterator = formatted_obj_recursive_iterator[
                        key
                    ]

            return formatted_obj

        return _filter_method
