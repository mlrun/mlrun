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


import mlrun
import mlrun.common.schemas

import services.api.utils.image_builder.base as image_builder
import services.api.utils.image_builder.buildah as buildah_image_builder
import services.api.utils.image_builder.kaniko as kaniko_image_builder


class ImageBuilderFactory:
    _builders: dict[
        mlrun.common.schemas.ContainerBuilderKind,
        type[image_builder.AbstractBaseImageBuilder],
    ] = {
        mlrun.common.schemas.ContainerBuilderKind.kaniko: kaniko_image_builder.KanikoImageBuilder,
        mlrun.common.schemas.ContainerBuilderKind.buildah: buildah_image_builder.BuildahImageBuilder,
    }

    @classmethod
    def create_builder(
        cls, kind: mlrun.common.schemas.ContainerBuilderKind | None = None
    ) -> image_builder.AbstractBaseImageBuilder:
        if kind is None:
            kind = (
                mlrun.mlconf.httpdb.builder.container_builder_kind
                or mlrun.common.schemas.ContainerBuilderKind.kaniko.value
            )
            kind = mlrun.common.schemas.ContainerBuilderKind(kind)
        return cls._builders[kind]()
