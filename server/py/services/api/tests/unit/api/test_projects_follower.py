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

import collections.abc
import unittest.mock
import uuid

import pytest

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton

import services.api.api.endpoints.projects_follower as projects_follower
import services.api.crud.projects as projects_crud


@pytest.fixture
def reset_projects_singleton() -> collections.abc.Iterator[None]:
    """Drop the Projects singleton so __init__ re-runs with the current mlconf."""
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)
    yield
    mlrun.utils.singleton.Singleton._instances.pop(projects_crud.Projects, None)


@pytest.mark.asyncio
async def test_get_project_state_returns_state_for_existing_project(
    reset_projects_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
):
    op_id = uuid.UUID(int=1)
    snapshot = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="proj"),
        status=mlrun.common.schemas.ProjectStatus(
            op_id=op_id, state=mlrun.common.schemas.ProjectState.online
        ),
    )
    monkeypatch.setattr(
        projects_crud.Projects,
        "get_follower_project_snapshot",
        lambda *a, **k: snapshot,
    )

    result = await projects_follower.get_project_state(
        "proj", unittest.mock.MagicMock()
    )

    assert result == projects_follower.follower_schemas.FollowerProjectState(
        name="proj",
        op_id=op_id,
        sync_status=mlrun.common.schemas.ProjectState.online,
    )


@pytest.mark.asyncio
async def test_get_project_state_raises_not_found_for_missing_project(
    reset_projects_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        projects_crud.Projects, "get_follower_project_snapshot", lambda *a, **k: None
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        await projects_follower.get_project_state("missing", unittest.mock.MagicMock())


@pytest.mark.asyncio
async def test_get_project_state_passes_session_and_name_to_the_snapshot_lookup(
    reset_projects_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def _fake_get_follower_project_snapshot(self, session, name):
        calls.append((session, name))
        return None

    monkeypatch.setattr(
        projects_crud.Projects,
        "get_follower_project_snapshot",
        _fake_get_follower_project_snapshot,
    )
    db_session = unittest.mock.MagicMock()

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        await projects_follower.get_project_state("proj", db_session)

    assert calls == [(db_session, "proj")]
