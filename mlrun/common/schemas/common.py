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

import typing

import pydantic


class ImageBuilder(pydantic.BaseModel):
    functionSourceCode: typing.Optional[str] = None  # noqa: N815
    codeEntryType: typing.Optional[str] = None  # noqa: N815
    codeEntryAttributes: typing.Optional[str] = None  # noqa: N815
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

    class Config:
        extra = pydantic.Extra.allow
