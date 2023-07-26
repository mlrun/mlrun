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

import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest

import mlrun.api.api.utils
import mlrun.api.launcher
import mlrun.common.schemas
import mlrun.launcher.base
import mlrun.launcher.factory


@pytest.mark.parametrize(
    "is_remote, local, expectation",
    [
        (True, False, does_not_raise()),
        (False, False, does_not_raise()),
        # local run is not allowed when running as API
        (True, True, pytest.raises(mlrun.errors.MLRunInternalServerError)),
        (False, True, pytest.raises(mlrun.errors.MLRunInternalServerError)),
    ],
)
def test_create_server_side_launcher(is_remote, local, expectation):
    """Test that the server side launcher is created when we are running as API"""
    with expectation:
        launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
            is_remote,
            local=local,
        )
        assert isinstance(launcher, mlrun.api.launcher.ServerSideLauncher)


def test_enrich_and_validate_with_auth_info():
    auth_info = mlrun.common.schemas.auth.AuthInfo(
        access_key="access_key",
        username="username",
    )
    launcher_kwargs = {"auth_info": auth_info}
    launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
        is_remote=True,
        **launcher_kwargs,
    )

    assert launcher._auth_info == auth_info
    function = mlrun.new_function(
        name="launcher-test",
        kind="job",
    )

    with unittest.mock.patch(
        "mlrun.api.api.utils.apply_enrichment_and_validation_on_function",
        unittest.mock.Mock(),
    ) as apply_enrichment_and_validation_on_function:

        launcher.enrich_runtime(function)
        apply_enrichment_and_validation_on_function.assert_called_once_with(
            function,
            auth_info,
        )
