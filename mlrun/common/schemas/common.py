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
import typing

import pydantic


class ImageBuilder(pydantic.BaseModel):
    functionSourceCode: typing.Optional[str] = None
    codeEntryType: typing.Optional[str] = None
    codeEntryAttributes: typing.Optional[str] = None
    source: typing.Optional[str] = None
    code_origin: typing.Optional[str] = None
    origin_filename: typing.Optional[str] = None
    image: typing.Optional[str] = None
    base_image: typing.Optional[str] = None
    commands: typing.Optional[list] = None
    extra: typing.Optional[str] = None
    extra_args: typing.Optional[dict] = None
    builder_env: typing.Optional[dict] = None
    secret: typing.Optional[str] = None
    registry: typing.Optional[str] = None
    load_source_on_run: typing.Optional[bool] = None
    with_mlrun: typing.Optional[bool] = None
    auto_build: typing.Optional[bool] = None
    build_pod: typing.Optional[str] = None
    requirements: typing.Optional[list] = None
    source_code_target_dir: typing.Optional[str] = None


class ObjectFormat:
    full = "full"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        return {
            ObjectFormat.full: None,
        }[_format]

    @classmethod
    def format_obj(cls, obj: dict, _format: str) -> dict:
        format_method = cls.format_method(_format)
        if not format_method:
            return obj

        return format_method(obj)

    @classmethod
    def filter_obj_method(cls, _filter: list[list[str]]) -> typing.Callable:
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
