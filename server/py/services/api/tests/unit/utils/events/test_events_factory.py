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
import services.api.utils.events.base
import services.api.utils.events.events_factory
import services.api.utils.events.iguazio
import services.api.utils.events.nop


@pytest.mark.parametrize(
    "events_mode,kind,igz_version,expected_error,expected_instance",
    [
        (
            mlrun.common.schemas.EventsModes.disabled,
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
            services.api.utils.events.nop.NopClient,
        ),
        (
            mlrun.common.schemas.EventsModes.enabled,
            mlrun.common.schemas.EventClientKinds.iguazio,
            None,
            mlrun.errors.MLRunInvalidArgumentError,
            None,
        ),
        (
            mlrun.common.schemas.EventsModes.enabled,
            mlrun.common.schemas.EventClientKinds.iguazio,
            "3.5.3",
            None,
            services.api.utils.events.iguazio.Client,
        ),
    ],
)
def test_get_events_client(
    events_mode: mlrun.common.schemas.EventsModes,
    kind: mlrun.common.schemas.EventClientKinds,
    igz_version: str,
    expected_error: mlrun.errors.MLRunBaseError,
    expected_instance: services.api.utils.events.base.BaseEventClient,
):
    mlrun.mlconf.events.mode = events_mode.value
    mlrun.mlconf.igz_version = igz_version
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
