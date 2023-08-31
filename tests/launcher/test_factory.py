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

import pytest

import mlrun.api.launcher
import mlrun.launcher.base
import mlrun.launcher.factory
import mlrun.launcher.local
import mlrun.launcher.remote


@pytest.mark.parametrize(
    "is_remote, local, expected_instance",
    [
        # runtime is remote and user didn't specify local - submit job flow
        (
            True,
            False,
            mlrun.launcher.remote.ClientRemoteLauncher,
        ),
        # runtime is remote but specify local - run local flow
        (
            True,
            True,
            mlrun.launcher.local.ClientLocalLauncher,
        ),
        # runtime is local and user specify local - run local flow
        (
            False,
            True,
            mlrun.launcher.local.ClientLocalLauncher,
        ),
        # runtime is local and user didn't specify local - run local flow
        (
            False,
            False,
            mlrun.launcher.local.ClientLocalLauncher,
        ),
    ],
)
def test_create_client_launcher(
    is_remote: bool,
    local: bool,
    expected_instance: typing.Union[
        mlrun.launcher.base.BaseLauncher,
        mlrun.launcher.local.ClientLocalLauncher,
    ],
):
    launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
        is_remote, local=local
    )
    assert type(launcher) is expected_instance

    if local:
        assert launcher._is_run_local

    elif not is_remote:
        assert not launcher._is_run_local
