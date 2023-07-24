import pytest

from mlrun.api.crud.model_monitoring.deployment import Minutes, Seconds, seconds2minutes


@pytest.mark.parametrize(
    ("seconds", "expected_minutes"),
    [(0, 0), (1, 1), (60, 1), (21, 1), (365, 7)],
)
def test_minutes2seconds(seconds, expected_minutes):
    assert seconds2minutes(seconds) == expected_minutes
