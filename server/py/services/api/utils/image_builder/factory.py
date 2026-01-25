# Copyright 2026 Iguazio
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


import mlrun.common.schemas
from mlrun.config import config

from services.api.utils.image_builder.base import AbstractBaseImageBuilder
from services.api.utils.image_builder.buildah import BuildahImageBuilder
from services.api.utils.image_builder.kaniko import KanikoImageBuilder


class ImageBuilderFactory:
    _builders: dict[
        mlrun.common.schemas.ContainerBuilderKind, type[AbstractBaseImageBuilder]
    ] = {
        mlrun.common.schemas.ContainerBuilderKind.kaniko: KanikoImageBuilder,
        mlrun.common.schemas.ContainerBuilderKind.buildah: BuildahImageBuilder,
    }

    @classmethod
    def create_builder(
        cls, kind: mlrun.common.schemas.ContainerBuilderKind | None = None
    ) -> AbstractBaseImageBuilder:
        if kind is None:
            kind = (
                config.httpdb.builder.container_builder_kind
                or mlrun.common.schemas.ContainerBuilderKind.kaniko.value
            )
            kind = mlrun.common.schemas.ContainerBuilderKind(kind)
        return cls._builders[kind]()
