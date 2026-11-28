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

import json
import uuid

import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors

import framework.utils.projects.follower_schemas as follower_schemas
import services.api.api.endpoints.projects_follower as projects_follower
import services.api.crud


@pytest.fixture
def follower_db_session(
    db: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> sqlalchemy.orm.Session:
    monkeypatch.setattr("framework.db.session.create_session", lambda: db)
    monkeypatch.setattr("framework.db.session.close_session", lambda session: None)
    return db


@pytest.mark.asyncio
async def test_get_project_state_returns_state_for_existing_project(
    db: sqlalchemy.orm.Session,
):
    op_id = uuid.uuid4()
    services.api.crud.Projects().create_project(
        db,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="proj"),
            status=mlrun.common.schemas.ProjectStatus(
                op_id=op_id, state=mlrun.common.schemas.ProjectState.online
            ),
        ),
    )

    result = await projects_follower.get_project_state("proj", db)

    assert result == projects_follower.follower_schemas.FollowerProjectState(
        name="proj",
        op_id=op_id,
        sync_status=mlrun.common.schemas.ProjectState.online,
    )


@pytest.mark.asyncio
async def test_get_project_state_raises_not_found_for_missing_project(
    db: sqlalchemy.orm.Session,
):
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        await projects_follower.get_project_state("missing", db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_state, sent_spec, expected_state",
    [
        ("online", {"desired_state": "archived"}, "archived"),
        ("archived", {"desired_state": "online"}, "online"),
        ("archived", {"desired_state": "archived"}, "archived"),
        # An older leader omits the field; the model default (online) must not be
        # mistaken for an instruction to un-archive.
        ("archived", {}, "archived"),
        ("archived", {"desired_state": None}, "archived"),
        ("online", {}, "online"),
    ],
    ids=[
        "archive",
        "unarchive",
        "archive-again",
        "absent-keeps-archived",
        "null-keeps-archived",
        "absent-keeps-online",
    ],
)
async def test_update_project_applies_desired_state(
    follower_db_session: sqlalchemy.orm.Session,
    stored_state: str,
    sent_spec: dict,
    expected_state: str,
):
    db = follower_db_session
    stored_op_id = uuid.UUID(int=1)
    new_op_id = uuid.UUID(int=2)
    _create_project(db, stored_state, stored_op_id)

    result = await projects_follower.update_project(
        "proj", _update_request(sent_spec, new_op_id, stored_op_id)
    )

    assert result.sync_status == expected_state
    assert result.op_id == new_op_id
    reported = await projects_follower.get_project_state("proj", db)
    assert reported.sync_status == expected_state
    stored = services.api.crud.Projects().get_project(db, "proj")
    assert stored.status.state == expected_state
    assert stored.spec.desired_state == expected_state


@pytest.mark.asyncio
async def test_update_project_rejects_offline_desired_state(
    follower_db_session: sqlalchemy.orm.Session,
):
    db = follower_db_session
    stored_op_id = uuid.UUID(int=1)
    _create_project(db, "online", stored_op_id)

    with pytest.raises(mlrun.errors.MLRunBadRequestError):
        await projects_follower.update_project(
            "proj",
            _update_request(
                {"desired_state": "offline"}, uuid.UUID(int=2), stored_op_id
            ),
        )

    reported = await projects_follower.get_project_state("proj", db)
    assert reported.sync_status == mlrun.common.schemas.ProjectState.online
    assert reported.op_id == stored_op_id


@pytest.mark.asyncio
async def test_update_project_with_stale_cas_does_not_apply_desired_state(
    follower_db_session: sqlalchemy.orm.Session,
):
    db = follower_db_session
    stored_op_id = uuid.UUID(int=5)
    _create_project(db, "online", stored_op_id)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        await projects_follower.update_project(
            "proj",
            _update_request(
                {"desired_state": "archived"}, uuid.UUID(int=6), uuid.UUID(int=4)
            ),
        )

    reported = await projects_follower.get_project_state("proj", db)
    assert reported.sync_status == mlrun.common.schemas.ProjectState.online
    assert reported.op_id == stored_op_id


@pytest.mark.asyncio
async def test_update_project_replay_does_not_apply_desired_state(
    follower_db_session: sqlalchemy.orm.Session,
):
    db = follower_db_session
    op_id = uuid.UUID(int=5)
    _create_project(db, "online", op_id)

    # Same op_id as stored is a replay: nothing is applied, even with a new state.
    result = await projects_follower.update_project(
        "proj", _update_request({"desired_state": "archived"}, op_id, uuid.UUID(int=4))
    )

    assert result.sync_status == mlrun.common.schemas.ProjectState.online
    reported = await projects_follower.get_project_state("proj", db)
    assert reported.sync_status == mlrun.common.schemas.ProjectState.online


def _create_project(db: sqlalchemy.orm.Session, state: str, op_id: uuid.UUID) -> None:
    services.api.crud.Projects().create_project(
        db,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="proj"),
            spec=mlrun.common.schemas.ProjectSpec(desired_state=state),
            status=mlrun.common.schemas.ProjectStatus(state=state, op_id=op_id),
        ),
    )


def _update_request(
    spec: dict, op_id: uuid.UUID, prev_op_id: uuid.UUID
) -> follower_schemas.FollowerUpdateRequest:
    # Parsed from JSON, like the wire body, so model_fields_set reflects what was sent.
    return follower_schemas.FollowerUpdateRequest.model_validate_json(
        json.dumps(
            {
                "metadata": {"name": "proj"},
                "spec": spec,
                "status": {"op_id": str(op_id)},
                "prev_op_id": str(prev_op_id),
            }
        )
    )
