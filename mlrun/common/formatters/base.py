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
    full = "full"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        return {
            ObjectFormat.full: None,
        }[_format]

    @classmethod
    def format_obj(
        cls,
        obj: typing.Any,
        _format: str,
        exclude_formats: typing.Optional[list[str]] = None,
    ) -> typing.Any:
        _format = _format or cls.full
        invalid_format_exc = mlrun.errors.MLRunBadRequestError(
            f"Provided format is not supported. format={_format}"
        )

        if _format in exclude_formats:
            raise invalid_format_exc

        try:
            format_method = cls.format_method(_format)
        except KeyError:
            raise invalid_format_exc
        if not format_method:
            return obj

        return format_method(obj)

    @staticmethod
    def filter_obj_method(_filter: list[list[str]]) -> typing.Callable:
        def _filter_method(obj: dict) -> dict:
            formatted_obj = {}
            for key_list in _filter:
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
