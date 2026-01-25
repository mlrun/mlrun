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


import pytest

import mlrun

import services.api.utils.image_builder.buildah as buildah_image_builder
import services.api.utils.image_builder.factory as image_builder_factory
import services.api.utils.image_builder.kaniko as kaniko_image_builder


@pytest.mark.parametrize(
    "kind_config,kind_param,expected_cls",
    [
        ("kaniko", None, kaniko_image_builder.KanikoImageBuilder),
        ("buildah", None, buildah_image_builder.BuildahImageBuilder),
        ("kaniko", "buildah", buildah_image_builder.BuildahImageBuilder),
        (None, None, kaniko_image_builder.KanikoImageBuilder),
        (None, "buildah", buildah_image_builder.BuildahImageBuilder),
    ],
)
def test_image_builder_factory_create_builder(
    kind_config, kind_param, expected_cls, monkeypatch
):
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.builder, "container_builder_kind", kind_config
    )
    builder = image_builder_factory.ImageBuilderFactory.create_builder(kind=kind_param)
    assert isinstance(builder, expected_cls)
