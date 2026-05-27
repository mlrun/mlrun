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

import collections
import collections.abc
import datetime
import types
import unittest.mock

import pytest

import mlrun
import mlrun.utils.singleton

import services.api.crud.projects as projects_crud


def _make_projects_output(project_names: list[str]) -> types.SimpleNamespace:
    """Build a list_projects() return stub for name_and_creation_time format."""
    now = datetime.datetime.now(tz=datetime.UTC)
    return types.SimpleNamespace(projects=[(name, now) for name in project_names])


def _make_project_counters(project_names: list[str]) -> tuple[dict, ...]:
    """20-tuple of {project: count} dicts matching get_project_resources_counters."""
    return tuple({name: 0 for name in project_names} for _ in range(20))


def _make_pipeline_counters() -> tuple[collections.defaultdict, ...]:
    """3-tuple of defaultdicts matching _calculate_pipelines_counters."""
    return tuple(collections.defaultdict(lambda: 0) for _ in range(3))


@pytest.fixture
def reset_projects_singleton() -> collections.abc.Iterator[None]:
    """Drop the Projects singleton so __init__ re-runs with the current mlconf."""
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)
    yield
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)


@pytest.fixture
def patched_refresh_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> unittest.mock.MagicMock:
    """Stub out everything refresh_project_resources_counters_cache touches.

    Returns the mock for ``telemetry_inventory.set_count`` so the caller can
    assert on its invocation count.
    """
    project_names = ["proj-a", "proj-b"]
    projects_output = _make_projects_output(project_names)
    project_counters = _make_project_counters(project_names)
    pipeline_counters = _make_pipeline_counters()

    # list_projects is wrapped in run_function_with_new_db_session — patch the
    # session helper to call its target directly with the supplied kwargs.
    def _run_sync(func, *args, **kwargs):
        return func(unittest.mock.MagicMock(), *args, **kwargs)

    monkeypatch.setattr(
        "framework.db.session.run_function_with_new_db_session", _run_sync
    )

    db_mock = unittest.mock.MagicMock()
    db_mock.get_project_resources_counters = unittest.mock.AsyncMock(
        return_value=project_counters
    )
    db_mock.refresh_project_summaries = unittest.mock.MagicMock()
    monkeypatch.setattr("framework.utils.singletons.db.get_db", lambda: db_mock)

    # list_projects is a sync method on Projects — patch the class-level method
    # so it returns our stub regardless of arguments.
    monkeypatch.setattr(
        projects_crud.Projects,
        "list_projects",
        lambda self, *args, **kwargs: projects_output,
    )
    monkeypatch.setattr(
        projects_crud.Projects,
        "_calculate_pipelines_counters",
        unittest.mock.AsyncMock(return_value=pipeline_counters),
    )

    set_count_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(projects_crud.telemetry_inventory, "set_count", set_count_mock)
    # The real SDK isn't initialized in tests, so force the enabled flag on.
    monkeypatch.setattr(projects_crud.telemetry_inventory, "is_enabled", lambda: True)
    return set_count_mock


@pytest.mark.asyncio
async def test_inventory_telemetry_emits_every_cycle_when_multiplier_is_one(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """multiplier=1 → emission fires on every cache refresh."""
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 1
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    assert crud._inventory_emit_multiplier == 1

    for _ in range(5):
        await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    # mlrun_projects is emitted exactly once per refresh, so call count for it
    # is a clean proxy for "how many refreshes emitted".
    projects_calls = [
        c for c in set_count_mock.call_args_list if c.args[0] == "mlrun_projects"
    ]
    assert len(projects_calls) == 5


@pytest.mark.asyncio
async def test_inventory_telemetry_emits_every_nth_cycle(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """multiplier=3 → emission fires on refreshes 0, 3, 6 (1-in-3 cycles)."""
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 3
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    assert crud._inventory_emit_multiplier == 3

    for _ in range(7):
        await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    projects_calls = [
        c for c in set_count_mock.call_args_list if c.args[0] == "mlrun_projects"
    ]
    # iterations 0, 3, 6 → 3 emissions
    assert len(projects_calls) == 3


@pytest.mark.asyncio
async def test_inventory_emit_multiplier_clamps_invalid_config(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """multiplier=0 (invalid) must clamp to 1, so emission still fires every cycle."""
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 0
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    assert crud._inventory_emit_multiplier == 1

    for _ in range(3):
        await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    projects_calls = [
        c for c in set_count_mock.call_args_list if c.args[0] == "mlrun_projects"
    ]
    assert len(projects_calls) == 3


@pytest.mark.asyncio
async def test_inventory_telemetry_caches_multiplier_at_init(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """Live edits to mlconf after singleton init must not change the cadence."""
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 2
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    assert crud._inventory_emit_multiplier == 2

    # Change config after construction — should be ignored until restart.
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 1

    for _ in range(4):
        await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    projects_calls = [
        c for c in set_count_mock.call_args_list if c.args[0] == "mlrun_projects"
    ]
    # iterations 0, 2 → 2 emissions (still cadence-2, ignoring the new config)
    assert len(projects_calls) == 2


@pytest.mark.asyncio
async def test_inventory_emits_every_registered_metric(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """Every metric in inventory._METRIC_NAMES must be emitted at least once.

    Regression guard: adding a name to ``_METRIC_NAMES`` without wiring a
    corresponding ``set_count`` call (or vice versa) silently produces an
    empty Prometheus series. This test catches both drift directions.
    """
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 1
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    emitted = {call.args[0] for call in set_count_mock.call_args_list}
    expected = set(projects_crud.telemetry_inventory._METRIC_NAMES)
    assert emitted == expected, (
        f"missing from emission: {expected - emitted}; "
        f"emitted but not registered: {emitted - expected}"
    )


@pytest.mark.asyncio
async def test_inventory_emission_tags_every_call_with_project(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
) -> None:
    """Every per-project metric must carry a ``project`` attribute.

    Only ``mlrun_projects`` is a system-level total; all others must be
    scoped per project so Prometheus can aggregate by project.
    """
    mlrun.mlconf.telemetry.system_counters.export_interval_multiplier = 1
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    for call in set_count_mock.call_args_list:
        metric_name = call.args[0]
        if metric_name == "mlrun_projects":
            continue
        assert "project" in call.kwargs, (
            f"{metric_name} emitted without project attribute: {call.kwargs}"
        )


@pytest.mark.asyncio
async def test_inventory_emission_skipped_when_telemetry_disabled(
    reset_projects_singleton: None,
    patched_refresh_dependencies: unittest.mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_enabled()=False at __init__ → no emission, even on a refresh cycle."""
    monkeypatch.setattr(projects_crud.telemetry_inventory, "is_enabled", lambda: False)
    set_count_mock = patched_refresh_dependencies

    crud = projects_crud.Projects()
    assert crud._inventory_telemetry_enabled is False

    await crud.refresh_project_resources_counters_cache(unittest.mock.MagicMock())

    set_count_mock.assert_not_called()
