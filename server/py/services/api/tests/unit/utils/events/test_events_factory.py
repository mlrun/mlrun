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

import pytest

import mlrun.common.schemas
import mlrun.common.types

import services.api.utils.events.base
import services.api.utils.events.events_factory
import services.api.utils.events.iguazio
import services.api.utils.events.iguazio_v4
import services.api.utils.events.nop


@pytest.mark.parametrize(
    "events_mode,kind,igz_version,auth_mode,expected_error,expected_instance",
    [
        (
            mlrun.common.schemas.EventsModes.disabled,
            None,
            None,
            None,
            None,
            services.api.utils.events.nop.NopClient,
        ),
        (
            mlrun.common.schemas.EventsModes.enabled,
            None,
            None,
            None,
            None,
            services.api.utils.events.nop.NopClient,
        ),
        (
            mlrun.common.schemas.EventsModes.enabled,
            mlrun.common.schemas.EventClientKinds.iguazio,
            None,
            None,
            mlrun.errors.MLRunInvalidArgumentError,
            None,
        ),
        (
            mlrun.common.schemas.EventsModes.enabled,
            mlrun.common.schemas.EventClientKinds.iguazio,
            "3.5.3",
            None,
            None,
            services.api.utils.events.iguazio.Client,
        ),
        # v4 mode auto-selected when auth mode is iguazio-v4
        (
            mlrun.common.schemas.EventsModes.enabled,
            None,
            "4.0.0",
            mlrun.common.types.AuthenticationMode.IGUAZIO_V4,
            None,
            services.api.utils.events.iguazio_v4.Client,
        ),
        # explicit v4 kind without v4 auth mode -> error
        (
            mlrun.common.schemas.EventsModes.enabled,
            mlrun.common.schemas.EventClientKinds.iguazio_v4,
            "3.5.3",
            None,
            mlrun.errors.MLRunInvalidArgumentError,
            None,
        ),
    ],
)
def test_get_events_client(
    events_mode: mlrun.common.schemas.EventsModes,
    kind: mlrun.common.schemas.EventClientKinds,
    igz_version: str,
    auth_mode: mlrun.common.types.AuthenticationMode,
    expected_error: mlrun.errors.MLRunBaseError,
    expected_instance: services.api.utils.events.base.BaseEventClient,
):
    mlrun.mlconf.events.mode = events_mode.value
    mlrun.mlconf.igz_version = igz_version
    mlrun.mlconf.httpdb.authentication.mode = (
        auth_mode.value
        if auth_mode
        else mlrun.common.types.AuthenticationMode.NONE.value
    )
    if expected_error:
        with pytest.raises(expected_error):
            services.api.utils.events.events_factory.EventsFactory.get_events_client(
                kind
            )
    else:
        instance = (
            services.api.utils.events.events_factory.EventsFactory.get_events_client(
                kind
            )
        )
        assert isinstance(instance, expected_instance)
