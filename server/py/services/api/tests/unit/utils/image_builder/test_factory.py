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

from mlrun.config import config

from services.api.utils.image_builder.buildah import BuildahImageBuilder
from services.api.utils.image_builder.factory import ImageBuilderFactory
from services.api.utils.image_builder.kaniko import KanikoImageBuilder


@pytest.mark.parametrize(
    "kind_config,kind_param,expected_cls",
    [
        ("kaniko", None, KanikoImageBuilder),
        ("buildah", None, BuildahImageBuilder),
        ("kaniko", "buildah", BuildahImageBuilder),
        (None, None, KanikoImageBuilder),
        (None, "buildah", BuildahImageBuilder),
    ],
)
def test_image_builder_factory_create_builder(
    kind_config, kind_param, expected_cls, monkeypatch
):
    monkeypatch.setattr(config.httpdb.builder, "container_builder_kind", kind_config)
    builder = ImageBuilderFactory.create_builder(kind=kind_param)
    assert isinstance(builder, expected_cls)
