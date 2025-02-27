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
import time
import uuid

import sqlalchemy.orm

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
