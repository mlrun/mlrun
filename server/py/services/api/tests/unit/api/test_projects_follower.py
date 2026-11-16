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

import uuid

import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.errors

import services.api.api.endpoints.projects_follower as projects_follower
import services.api.crud


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
