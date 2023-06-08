import pytest

import mlrun.api.utils.events.base
import mlrun.api.utils.events.events_factory
import mlrun.api.utils.events.iguazio
import mlrun.api.utils.events.nop
import mlrun.common.schemas


@pytest.mark.parametrize(
    "events_mode,kind,igz_version,expected_error,expected_instance",
    [
        (
            mlrun.common.schemas.EventsMode.disabled,
            None,
            None,
            None,
            mlrun.api.utils.events.nop.NopClient,
        ),
        (
            mlrun.common.schemas.EventsMode.enabled,
            None,
            None,
            None,
            mlrun.api.utils.events.nop.NopClient,
        ),
        (
            mlrun.common.schemas.EventsMode.enabled,
            mlrun.common.schemas.EventClientsTypes.iguazio,
            None,
            mlrun.errors.MLRunInvalidArgumentError,
            None,
        ),
        (
            mlrun.common.schemas.EventsMode.enabled,
            mlrun.common.schemas.EventClientsTypes.iguazio,
            "3.5.3",
            None,
            mlrun.api.utils.events.iguazio.Client,
        ),
    ],
)
def test_get_events_client(
    events_mode: mlrun.common.schemas.EventsMode,
    kind: mlrun.common.schemas.EventClientsTypes,
    igz_version: str,
    expected_error: mlrun.errors.MLRunBaseError,
    expected_instance: mlrun.api.utils.events.base.BaseEventClient,
):
    mlrun.mlconf.events.mode = events_mode.value
    mlrun.mlconf.igz_version = igz_version
    if expected_error:
        with pytest.raises(expected_error):
            mlrun.api.utils.events.events_factory.EventsFactory.get_events_client(kind)
    else:
        instance = (
            mlrun.api.utils.events.events_factory.EventsFactory.get_events_client(kind)
        )
        assert isinstance(instance, expected_instance)
