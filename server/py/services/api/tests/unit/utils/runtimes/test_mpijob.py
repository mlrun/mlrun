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
import mlrun.errors
from mlrun.common.types import AuthenticationMode

import framework.utils.runtimes.mpijob


@pytest.mark.parametrize(
    "auth_mode",
    [
        AuthenticationMode.NONE,
        AuthenticationMode.BASIC,
        AuthenticationMode.BEARER,
        AuthenticationMode.IGUAZIO,
    ],
)
def test_validate_mpijob_runtime_supported_passes_when_not_iguazio_v4(
    monkeypatch, auth_mode
):
    monkeypatch.setattr(mlrun.mlconf.httpdb.authentication, "mode", auth_mode)

    # MPIJob is supported outside IG4
    framework.utils.runtimes.mpijob.validate_mpijob_runtime_supported()


def test_validate_mpijob_runtime_supported_raises_in_iguazio_v4(monkeypatch):
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication, "mode", AuthenticationMode.IGUAZIO_V4
    )

    with pytest.raises(mlrun.errors.MLRunBadRequestError) as exc_info:
        framework.utils.runtimes.mpijob.validate_mpijob_runtime_supported()

    assert framework.utils.runtimes.mpijob.mpijob_runtime_unsupported_message in str(
        exc_info.value
    )
