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
import pytest

import mlrun.common.schemas
import mlrun.utils.capabilities


class TestCapabilities:
    @pytest.mark.parametrize(
        "namespace,mode,expected",
        [
            ("", None, False),
            ("mlrun", None, True),
            ("mlrun", mlrun.common.schemas.CapabilitiesModes.enabled.value, True),
        ],
    )
    def test_k8s(self, namespace, mode, expected):
        mlrun.mlconf.namespace = namespace
        mlrun.mlconf.capabilities.k8s.mode = mode
        assert mlrun.utils.capabilities.Capabilities.k8s() == expected
