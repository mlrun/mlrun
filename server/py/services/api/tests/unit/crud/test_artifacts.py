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

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.common.schemas.artifact

import services.api.crud


class TestArtifacts:
    def test_list_artifacts(
        self,
        db: sqlalchemy.orm.Session,
    ):
        tree, key = "tree", "key"
        project = "project-name"
        artifact = self._generate_artifact(project, tree, key)
        services.api.crud.Artifacts().store_artifact(
            db,
            artifact["spec"]["db_key"],
            artifact,
            project=project,
        )
        artifacts = services.api.crud.Artifacts().list_artifacts(
            db, project, tag="*", limit=100
        )
        assert len(artifacts) == 1, "bad number of artifacts"

        artifact_kinds = [
            artifact_category.value
            for artifact_category in mlrun.common.schemas.artifact.ArtifactCategories.all()
        ]
        for artifact_kind in artifact_kinds:
            artifact = self._generate_artifact(project, tree, key, kind=artifact_kind)
            time.sleep(0.01)
            services.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                project=project,
            )

        expected_length = len(artifact_kinds) + 1
        artifacts = services.api.crud.Artifacts().list_artifacts(db, project, tag="*")
        assert len(artifacts) == expected_length, "bad number of artifacts"

        # validate ordering by checking that the first artifact is the latest one
        assert artifacts[0]["kind"] == artifact_kinds[-1], "bad ordering"

        # validate ordering by checking that list of returned artifacts is sorted
        # by updated time in descending order
        for i in range(1, len(artifacts)):
            assert (
                artifacts[i]["metadata"]["updated"]
                <= artifacts[i - 1]["metadata"]["updated"]
            ), "bad ordering"

    def test_store_and_get_artifact_missing_project(self, db: sqlalchemy.orm.Session):
        key = "artifact-key"
        tree = "artifact-tree"
        project = "project-name"

        artifact = self._generate_artifact(project, tree, key)

        # store with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            services.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                project=None,
            )

        # store with valid project
        services.api.crud.Artifacts().store_artifact(
            db,
            artifact["spec"]["db_key"],
            artifact,
            project=project,
        )

        # get with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            services.api.crud.Artifacts().get_artifact(
                db, key=key, tag="latest", project=None
            )

        # list with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            services.api.crud.Artifacts().list_artifacts(db, project=None, tag="*")

    @pytest.mark.parametrize(
        "artifact_initial,auth_username,expected_producer",
        [
            # No producer: set default producer with owner and kind "api"
            (
                {"spec": {"db_key": "my-key"}},
                "user1",
                {"owner": "user1", "kind": "api"},
            ),
            # Producer without owner: set owner to authenticated user, keep kind
            (
                {"spec": {"producer": {"kind": "job"}}},
                "user3",
                {"owner": "user3", "kind": "job"},
            ),
            # Same owner as authenticated user: leave producer unchanged
            (
                {"spec": {"producer": {"owner": "user4", "kind": "run"}}},
                "user4",
                {"owner": "user4", "kind": "run"},
            ),
            # Different owner: override with authenticated user
            (
                {"spec": {"producer": {"owner": "other-user", "kind": "run"}}},
                "user5",
                {"owner": "user5", "kind": "run"},
            ),
            # No auth_info: do not add or change producer
            ({"spec": {"db_key": "my-key"}}, None, None),
        ],
    )
    def test_enrich_artifact_producer(
        self, artifact_initial, auth_username, expected_producer
    ):
        artifacts_crud = services.api.crud.Artifacts()
        artifact = artifact_initial
        auth_info = (
            mlrun.common.schemas.AuthInfo(username=auth_username)
            if auth_username is not None
            else None
        )

        artifacts_crud._enrich_artifact_producer(artifact, auth_info)

        if expected_producer is None:
            assert "producer" not in artifact.get("spec", {})
        else:
            assert artifact["spec"]["producer"] == expected_producer

    @staticmethod
    def _generate_artifact(
        project,
        tree,
        key,
        kind="artifact",
        iter=None,
    ):
        artifact = {
            "kind": kind,
            "metadata": {
                "key": key,
                "tree": tree,
                "uid": str(uuid.uuid4()),
                "project": project,
                "iter": iter or 0,
                "tag": "latest",
            },
            "spec": {
                "db_key": key,
            },
            "status": {},
        }
        return artifact


def test_project_param_not_overwritten_by_artifact_uri():
    """
    Verify that the `project` parameter passed to `_get_artifacts_from_uris` is
    preserved and used for the final `list_artifacts_for_producer_id` call, even
    when artifact URIs contain a different project name.

    Before the fix, the tuple unpacking `project, key, ... = parse_artifact_uri(...)`
    shadowed the function's `project` parameter, causing `list_artifacts_for_producer_id`
    to be called with the project from the last artifact URI instead of the original.
    """
    original_project = "my-project"
    producer_id = "some-producer-uid"
    run = {
        "status": {
            "artifact_uris": {
                "result1": f"store://artifacts/{original_project}/key1@{producer_id}",
                "result2": f"store://artifacts/{original_project}/key2@{producer_id}",
            },
        },
    }
    db_session = MagicMock()

    with patch.object(
        services.api.crud.Artifacts,
        "list_artifacts_for_producer_id",
        return_value=[],
    ) as mock_list:
        services.api.crud.Runs._get_artifacts_from_uris(
            db_session, original_project, producer_id, run
        )

        # The project passed to list_artifacts_for_producer_id must be the original
        mock_list.assert_called_once()
        call_args = mock_list.call_args
        # project is the third positional argument (after db_session and producer_id)
        actual_project = call_args[0][2]
        assert actual_project == original_project, (
            f"Expected project '{original_project}' but got '{actual_project}'. "
            "The project parameter was overwritten by artifact URI parsing."
        )

def test_project_preserved_when_uri_has_different_project():
    """
    When an artifact URI contains a different project than the function parameter,
    the original project should still be used for the DB query.
    """
    original_project = "my-project"
    different_project = "other-project"
    producer_id = "some-uid"
    run = {
        "status": {
            "artifact_uris": {
                "art1": f"store://artifacts/{different_project}/key1@{producer_id}",
            },
        },
    }
    db_session = MagicMock()

    with patch.object(
        services.api.crud.Artifacts,
        "list_artifacts_for_producer_id",
        return_value=[],
    ) as mock_list:
        services.api.crud.Runs._get_artifacts_from_uris(
            db_session, original_project, producer_id, run
        )

        mock_list.assert_called_once()
        call_args = mock_list.call_args
        actual_project = call_args[0][2]
        assert actual_project == original_project, (
            f"Expected project '{original_project}' but got '{actual_project}'. "
            "The project parameter was overwritten by a URI with a different project."
        )
