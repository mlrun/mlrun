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
import copy
import tempfile

import deepdiff
import pytest
from sqlalchemy import distinct, select
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils
import server.api.db.sqldb.models
import server.api.initial_data
from mlrun.artifacts.base import LinkArtifact
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact
from mlrun.common.schemas.artifact import ArtifactCategories
from server.api.db.base import DBInterface


class TestArtifacts:
    def test_list_artifact_name_filter(self, db: DBInterface, db_session: Session):
        artifact_name_1 = "artifact_name_1"
        artifact_name_2 = "artifact_name_2"
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(artifact_name_1, tree=tree)
        artifact_2 = self._generate_artifact(artifact_name_2, tree=tree)

        db.store_artifact(
            db_session,
            artifact_name_1,
            artifact_1,
        )
        db.store_artifact(
            db_session,
            artifact_name_2,
            artifact_2,
        )
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 2

        artifacts = db.list_artifacts(db_session, name=artifact_name_1)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = db.list_artifacts(db_session, name=artifact_name_2)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

        artifacts = db.list_artifacts(db_session, name="~artifact_name")
        assert len(artifacts) == 2

    def test_list_artifact_iter_parameter(self, db: DBInterface, db_session: Session):
        artifact_name_1 = "artifact_name_1"
        artifact_name_2 = "artifact_name_2"
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(artifact_name_1, tree=tree)
        artifact_2 = self._generate_artifact(artifact_name_2, tree=tree)

        # Use iters with multiple digits, to make sure filtering them via regex works
        test_iters = [0, 5, 9, 42, 219, 2102]
        for iter in test_iters:
            artifact_1["iter"] = artifact_2["iter"] = iter
            db.store_artifact(db_session, artifact_name_1, artifact_1, iter=iter)
            db.store_artifact(db_session, artifact_name_2, artifact_2, iter=iter)

        # No filter on iter. All are expected
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == len(test_iters) * 2

        # look for the artifact with the "latest" tag - should return all iterations
        artifacts = db.list_artifacts(db_session, name=artifact_name_1, tag="latest")
        assert len(artifacts) == len(test_iters)

        # Look for the various iteration numbers. Note that 0 is a special case due to the DB structure
        for iter in test_iters:
            artifacts = db.list_artifacts(db_session, iter=iter)
            assert len(artifacts) == 2
            for artifact in artifacts:
                assert artifact["iter"] == iter

        # Negative test
        artifacts = db.list_artifacts(db_session, iter=666)
        assert len(artifacts) == 0

        # Iter filter and a name filter, make sure query composition works
        artifacts = db.list_artifacts(db_session, name=artifact_name_1, iter=2102)
        assert len(artifacts) == 1

    def test_list_artifact_kind_filter(self, db: DBInterface, db_session: Session):
        artifact_name_1 = "artifact_name_1"
        artifact_kind_1 = ChartArtifact.kind
        artifact_name_2 = "artifact_name_2"
        artifact_kind_2 = PlotArtifact.kind
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(
            artifact_name_1, kind=artifact_kind_1, tree=tree
        )
        artifact_2 = self._generate_artifact(
            artifact_name_2, kind=artifact_kind_2, tree=tree
        )

        db.store_artifact(
            db_session,
            artifact_name_1,
            artifact_1,
        )
        db.store_artifact(
            db_session,
            artifact_name_2,
            artifact_2,
        )
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 2

        artifacts = db.list_artifacts(db_session, kind=artifact_kind_1)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = db.list_artifacts(db_session, kind=artifact_kind_2)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

    def test_list_artifact_category_filter(self, db: DBInterface, db_session: Session):
        artifact_name_1 = "artifact_name_1"
        artifact_kind_1 = ChartArtifact.kind
        artifact_name_2 = "artifact_name_2"
        artifact_kind_2 = PlotArtifact.kind
        artifact_name_3 = "artifact_name_3"
        artifact_kind_3 = ModelArtifact.kind
        artifact_name_4 = "artifact_name_4"
        artifact_kind_4 = DatasetArtifact.kind
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(
            artifact_name_1, kind=artifact_kind_1, tree=tree
        )
        artifact_2 = self._generate_artifact(
            artifact_name_2, kind=artifact_kind_2, tree=tree
        )
        artifact_3 = self._generate_artifact(
            artifact_name_3, kind=artifact_kind_3, tree=tree
        )
        artifact_4 = self._generate_artifact(
            artifact_name_4, kind=artifact_kind_4, tree=tree
        )

        for artifact_name, artifact_object in [
            (artifact_name_1, artifact_1),
            (artifact_name_2, artifact_2),
            (artifact_name_3, artifact_3),
            (artifact_name_4, artifact_4),
        ]:
            db.store_artifact(
                db_session,
                artifact_name,
                artifact_object,
            )

        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 4

        artifacts = db.list_artifacts(
            db_session, category=mlrun.common.schemas.ArtifactCategories.model
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_3

        artifacts = db.list_artifacts(
            db_session, category=mlrun.common.schemas.ArtifactCategories.dataset
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_4

        artifacts = db.list_artifacts(
            db_session, category=mlrun.common.schemas.ArtifactCategories.other
        )
        assert len(artifacts) == 2
        assert artifacts[0]["metadata"]["key"] == artifact_name_1
        assert artifacts[1]["metadata"]["key"] == artifact_name_2

    def test_list_artifact_label_filter(self, db: DBInterface, db_session: Session):
        total_artifacts = 5
        for i in range(1, total_artifacts + 1):
            artifact_name = f"artifact_name_{i}"
            artifact_tree = f"tree_{i}"
            artifact_labels = {"same_key": "same_value", f"label_{i}": f"value_{i}"}
            artifact = self._generate_artifact(
                artifact_name, tree=artifact_tree, labels=artifact_labels
            )
            db.store_artifact(
                db_session,
                artifact_name,
                artifact,
            )

        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == total_artifacts

        artifacts = db.list_artifacts(db_session, labels="same_key=same_value")
        assert len(artifacts) == total_artifacts

        artifacts = db.list_artifacts(db_session, labels="same_key")
        assert len(artifacts) == total_artifacts

        artifacts = db.list_artifacts(db_session, labels="~label")
        assert len(artifacts) == total_artifacts

        artifacts = db.list_artifacts(db_session, labels="~LaBeL=~VALue")
        assert len(artifacts) == total_artifacts

        artifacts = db.list_artifacts(db_session, labels="label_1=~Value")
        assert len(artifacts) == 1

        artifacts = db.list_artifacts(db_session, labels="label_1=value_1")
        assert len(artifacts) == 1

        artifacts = db.list_artifacts(db_session, labels="label_1=value_2")
        assert len(artifacts) == 0

        artifacts = db.list_artifacts(db_session, labels="label_2=~VALUE_2")
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == "artifact_name_2"

    def test_store_artifact_tagging(self, db: DBInterface, db_session: Session):
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_tag = "artifact_tag_1"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_1_kind = ChartArtifact.kind
        artifact_1_with_kind_tree = "artifact_tree_2"
        artifact_2_tag = "artifact_tag_2"
        artifact_1_with_kind_body = self._generate_artifact(
            artifact_1_key, kind=artifact_1_kind, tree=artifact_1_with_kind_tree
        )

        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_with_kind_body,
            tag=artifact_2_tag,
        )
        artifact = db.read_artifact(db_session, artifact_1_key, tag=artifact_1_tag)
        assert artifact["kind"] == "artifact"
        artifact = db.read_artifact(
            db_session, artifact_1_key, tag="latest", raise_on_not_found=False
        )
        assert artifact is not None
        artifacts = db.list_artifacts(db_session, artifact_1_key, tag=artifact_2_tag)
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == artifact_1_kind
        artifacts = db.list_artifacts(db_session, artifact_1_key, tag="latest")
        assert len(artifacts) == 1

    def test_store_artifact_latest_tag(self, db: DBInterface, db_session: Session):
        project = "artifact_project"
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=project
        )
        artifact_2_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=project
        )
        artifact_1_body["spec"]["something"] = "same"
        artifact_2_body["spec"]["something"] = "different"

        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            project=project,
        )
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_2_body,
            project=project,
        )

        artifact_tags = db.list_artifact_tags(db_session, project)

        # make sure only a single "latest" tag is returned
        assert len(artifact_tags) == 1

        artifacts = db.list_artifacts(db_session, artifact_1_key, project=project)
        assert len(artifacts) == 2
        for artifact in artifacts:
            if artifact["metadata"].get("tag") == "latest":
                assert artifact["spec"]["something"] == "different"
            else:
                assert artifact["spec"]["something"] == "same"

    def test_store_artifact_restoring_multiple_tags(
        self, db: DBInterface, db_session: Session
    ):
        project = "artifact_project"
        artifact_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(
            artifact_key, tree=artifact_1_tree, project=project
        )
        artifact_2_body = self._generate_artifact(
            artifact_key, tree=artifact_2_tree, project=project
        )
        artifact_1_tag = "artifact-tag-1"
        artifact_2_tag = "artifact-tag-2"

        # we use deepcopy to avoid changing the original dict
        db.store_artifact(
            db_session,
            artifact_key,
            copy.deepcopy(artifact_1_body),
            tag=artifact_1_tag,
            project=project,
        )
        db.store_artifact(
            db_session,
            artifact_key,
            copy.deepcopy(artifact_2_body),
            tag=artifact_2_tag,
            project=project,
        )
        artifacts = db.list_artifacts(
            db_session, artifact_key, tag="*", project=project
        )
        assert len(artifacts) == 3  # latest is also returned

        # ids are auto generated using this util function
        expected_uids = [
            mlrun.artifacts.base.fill_artifact_object_hash(artifact_body)
            for artifact_body in [artifact_1_body, artifact_2_body]
        ]
        uids = [artifact["metadata"]["uid"] for artifact in artifacts]
        assert (
            deepdiff.DeepDiff(
                expected_uids,
                uids,
                ignore_order=True,
            )
            == {}
        )
        expected_tags = [artifact_1_tag, artifact_2_tag, "latest"]
        tags = [artifact["metadata"]["tag"] for artifact in artifacts]
        assert (
            deepdiff.DeepDiff(
                expected_tags,
                tags,
                ignore_order=True,
            )
            == {}
        )
        artifact = db.read_artifact(db_session, artifact_key, tag=artifact_1_tag)
        assert artifact["metadata"]["uid"] == expected_uids[0]
        assert artifact["metadata"]["tag"] == artifact_1_tag
        artifact = db.read_artifact(db_session, artifact_key, tag=artifact_2_tag)
        assert artifact["metadata"]["uid"] == expected_uids[1]
        assert artifact["metadata"]["tag"] == artifact_2_tag

    def test_store_artifact_with_different_labels(
        self, db: DBInterface, db_session: Session
    ):
        # create an artifact with a single label
        project = "artifact_project"
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=project
        )
        labels = {"label1": "value1"}
        artifact_1_body["metadata"]["labels"] = {"label1": "value1"}
        artifact_1_body_copy = copy.deepcopy(artifact_1_body)
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            project=project,
        )

        artifacts = db.list_artifacts(db_session, artifact_1_key, project=project)
        assert len(artifacts) == 1
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["updated"])
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["created"])

        # add a new label to the same artifact
        labels["label2"] = "value2"
        artifact_1_body_copy["metadata"]["labels"] = labels
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body_copy,
            project=project,
        )

        # verify that the artifact has both labels and it didn't create a new artifact
        artifacts = db.list_artifacts(db_session, artifact_1_key, project=project)
        assert len(artifacts) == 1
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["updated"])
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["created"])
        assert (
            deepdiff.DeepDiff(
                artifacts[0].get("metadata", {}).get("labels", {}),
                labels,
                ignore_order=True,
            )
            == {}
        )

    def test_store_artifact_replace_tag(self, db: DBInterface, db_session: Session):
        project = "artifact_project"
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=project
        )
        artifact_1_tag = "artifact-tag-1"

        artifact_1_uid = db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
            project=project,
        )

        # verify that the artifact has the tag
        artifacts = db.list_artifacts(
            db_session, artifact_1_key, project=project, tag=artifact_1_tag
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_1_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_1_tree

        # create a new artifact with the same key and tag, but a different tree
        artifact_2_tree = "artifact_tree_2"
        artifact_2_body = self._generate_artifact(
            artifact_1_key, tree=artifact_2_tree, project=project
        )

        artifact_2_uid = db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_2_body,
            tag=artifact_1_tag,
            project=project,
        )

        # verify that only the new artifact has the tag
        artifacts = db.list_artifacts(
            db_session, artifact_1_key, project=project, tag=artifact_1_tag
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_2_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_2_tree

        # verify that the old artifact is still there, but without the tag
        artifacts = db.list_artifacts(db_session, artifact_1_key, project=project)
        assert len(artifacts) == 3

    def test_store_artifact_with_different_key(
        self, db: DBInterface, db_session: Session
    ):
        artifact_key = "artifact_key"
        artifact_different_key = "artifact_different_key"
        artifact_tree = "artifact_tree"

        artifact_body = self._generate_artifact(artifact_key, tree=artifact_tree)
        db.store_artifact(
            db_session,
            artifact_different_key,
            artifact_body,
        )
        artifact = db.read_artifact(db_session, artifact_different_key)
        assert artifact
        assert artifact["metadata"]["key"] == artifact_key

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.read_artifact(db_session, artifact_key)

    def test_read_artifact_tag_resolution(self, db: DBInterface, db_session: Session):
        """
        We had a bug in which when we got a tag filter for read/list artifact, we were transforming this tag to list of
        possible uids which is wrong, since a different artifact might have this uid as well, and we will return it,
        although it's not really tag with the given tag
        """
        artifact_1_key = "artifact_key_1"
        artifact_2_key = "artifact_key_2"
        artifact_tree = "artifact_uid_1"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_tree)
        artifact_1_tag = "artifact-tag-1"
        artifact_2_tag = "artifact-tag-2"

        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        db.store_artifact(
            db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
        )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.read_artifact(db_session, artifact_1_key, tag=artifact_2_tag)
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.read_artifact(db_session, artifact_2_key, tag=artifact_1_tag)
        # just verifying it's not raising
        db.read_artifact(db_session, artifact_1_key, tag=artifact_1_tag)
        db.read_artifact(db_session, artifact_2_key, tag=artifact_2_tag)
        # check list
        artifacts = db.list_artifacts(db_session, tag=artifact_1_tag)
        assert len(artifacts) == 1
        artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
        assert len(artifacts) == 1

    def test_overwrite_artifact_with_tag(self, db: DBInterface, db_session: Session):
        project = "proj"
        artifact_key = "artifact_key"
        artifact_tree = "artifact_uid"
        artifact_tree_2 = "artifact_uid_2"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, kind=ArtifactCategories.model
        )
        artifact_body_2 = self._generate_artifact(
            artifact_key, tree=artifact_tree_2, kind=ArtifactCategories.model
        )
        artifact_1_tag = "artifact-tag-1"
        artifact_2_tag = "artifact-tag-2"

        db.store_artifact(
            db_session, artifact_key, artifact_body, tag=artifact_1_tag, project=project
        )
        db.store_artifact(
            db_session,
            artifact_key,
            artifact_body_2,
            tag=artifact_2_tag,
            project=project,
        )

        identifier_1 = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key,
            uid=artifact_tree,
            tag=artifact_1_tag,
        )

        # overwrite the tag for only one of the artifacts
        db.overwrite_artifacts_with_tag(db_session, project, "new-tag", [identifier_1])

        # verify that only the first artifact is with the new tag now
        artifacts = db.list_artifacts(db_session, project=project, tag="new-tag")
        assert len(artifacts) == 1
        artifacts = db.list_artifacts(db_session, project=project, tag=artifact_1_tag)
        assert len(artifacts) == 0

        # verify that the second artifact's tag did not change
        artifacts = db.list_artifacts(db_session, project=project, tag=artifact_2_tag)
        assert len(artifacts) == 1

    def test_delete_artifacts_tag_filter(self, db: DBInterface, db_session: Session):
        artifact_1_key = "artifact_key_1"
        artifact_2_key = "artifact_key_2"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_2_tree)
        artifact_1_tag = "artifact-tag-one"
        artifact_2_tag = "artifact-tag-two"

        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        db.store_artifact(
            db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
        )
        db.del_artifacts(db_session, tag=artifact_1_tag)
        artifacts = db.list_artifacts(db_session, tag=artifact_1_tag)
        assert len(artifacts) == 0
        artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
        assert len(artifacts) == 1
        db.del_artifacts(db_session, tag=artifact_2_tag)
        artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
        assert len(artifacts) == 0

    def test_delete_artifact_tag_filter(self, db: DBInterface, db_session: Session):
        project = "artifact_project"
        artifact_1_key = "artifact_key_1"
        artifact_2_key = "artifact_key_2"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_2_tree)
        artifact_1_tag = "artifact-tag-one"
        artifact_2_tag = "artifact-tag-two"
        artifact_2_tag_2 = "artifact-tag-two-again"

        for artifact_key, artifact_body, artifact_tag in [
            (artifact_1_key, artifact_1_body, artifact_1_tag),
            (artifact_2_key, artifact_2_body, artifact_2_tag),
            (artifact_2_key, artifact_2_body, artifact_2_tag_2),
        ]:
            # we copy the artifact body to avoid changing the original dict
            artifact = copy.deepcopy(artifact_body)
            db.store_artifact(
                db_session,
                artifact_key,
                artifact,
                tag=artifact_tag,
                project=project,
            )

        artifacts = db.list_artifacts(db_session, project=project, name=artifact_1_key)
        # Should return 2 tags ('latest' and artifact_1_tag)
        assert len(artifacts) == 2
        artifacts = db.list_artifacts(db_session, project=project, tag=artifact_2_tag)
        assert len(artifacts) == 1
        artifacts = db.list_artifacts(db_session, project=project, tag=artifact_2_tag_2)
        assert len(artifacts) == 1

        db.del_artifact(db_session, artifact_1_key, project=project, tag=artifact_1_tag)
        artifacts = db.list_artifacts(db_session, name=artifact_1_key)
        assert len(artifacts) == 0

        # Negative test - wrong tag, no deletions
        db.del_artifact(db_session, artifact_2_key, project=project, tag=artifact_1_tag)
        artifacts = db.list_artifacts(db_session, project=project, name=artifact_2_key)

        # Should return 3 tags ('latest' and artifact_2_tag and artifact_2_tag_2)
        assert len(artifacts) == 3
        assert (
            deepdiff.DeepDiff(
                [artifact["metadata"]["tag"] for artifact in artifacts],
                ["latest", artifact_2_tag, artifact_2_tag_2],
                ignore_order=True,
            )
            == {}
        )

        tags = db.list_artifact_tags(db_session, project)
        assert len(tags) == 3

        # Delete the artifact object (should delete all tags of the same artifact object)
        db.del_artifact(
            db_session, artifact_2_key, tag=artifact_2_tag_2, project=project
        )
        artifacts = db.list_artifacts(db_session, project=project, name=artifact_2_key)
        assert len(artifacts) == 0

        # Assert all tags were deleted
        tags = db.list_artifact_tags(db_session, project)
        assert len(tags) == 0

    def test_list_artifacts_exact_name_match(
        self, db: DBInterface, db_session: Session
    ):
        artifact_1_key = "pre_artifact_key_suffix"
        artifact_2_key = "pre-artifact-key-suffix"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_2_tree)

        # Store each twice - once with no iter, and once with an iter
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
        )
        artifact_1_body["iter"] = 42
        db.store_artifact(
            db_session,
            artifact_1_key,
            artifact_1_body,
            iter=42,
        )
        db.store_artifact(
            db_session,
            artifact_2_key,
            artifact_2_body,
        )
        artifact_2_body["iter"] = 42
        db.store_artifact(
            db_session,
            artifact_2_key,
            artifact_2_body,
            iter=42,
        )

        def _list_and_assert_count(key, count, iter=None):
            results = db.list_artifacts(db_session, name=key, iter=iter)
            assert len(results) == count
            return results

        # Ensure fuzzy query works, and we have everything we need
        _list_and_assert_count("~key", count=4)

        # Do an exact match with underscores in the name - must escape the _ do it doesn't do a like query
        list_results = _list_and_assert_count(artifact_1_key, count=2)
        for artifact in list_results:
            assert artifact["metadata"]["key"] == artifact_1_key

        _list_and_assert_count("%key%", count=0)
        # Verify we don't get artifacts whose name is "%-suffix" due to the like query used in the DB
        _list_and_assert_count("suffix", count=0)
        # This should also be filtered, since the prefix is "pre" which is 3 chars. There's a known caveat if
        # prefix is 1 or 2 chars long.
        _list_and_assert_count("artifact-key-suffix", count=0)

        _list_and_assert_count(artifact_1_key, iter=42, count=1)
        _list_and_assert_count("~key", iter=42, count=2)
        _list_and_assert_count("~key", iter=666, count=0)

    def test_list_artifacts_best_iter_with_tagged_iteration(
        self, db: DBInterface, db_session: Session
    ):
        artifact_key_1 = "artifact-1"
        artifact_key_2 = "artifact-2"
        artifact_tree_1 = "tree-1"
        artifact_tree_2 = "tree-2"
        num_iters = 3
        best_iter = 2
        project = "project1"
        tag = "mytag1"

        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_key_1,
            artifact_tree_1,
            num_iters,
            best_iter,
            ArtifactCategories.model,
            project=project,
        )

        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_key_2,
            artifact_tree_2,
            num_iters,
            best_iter,
            ArtifactCategories.model,
            project=project,
        )

        identifier_1 = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key_1,
            iter=best_iter,
        )
        identifier_2 = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key_2,
            iter=best_iter,
        )
        db.append_tag_to_artifacts(
            db_session, project, tag, [identifier_1, identifier_2]
        )
        results = db.list_artifacts(
            db_session, project=project, tag=tag, best_iteration=True
        )
        assert len(results) == 2

        for artifact in results:
            assert (
                artifact["metadata"]["tag"] == tag
                and artifact["spec"]["iter"] == best_iter
                and artifact["metadata"]["key"] in (artifact_key_1, artifact_key_2)
            )

    def test_list_artifacts_best_iter(self, db: DBInterface, db_session: Session):
        artifact_1_key = "artifact-1"
        artifact_1_tree = "tree-1"
        artifact_2_key = "artifact-2"
        artifact_2_tree = "tree-2"
        artifact_no_link_key = "single-artifact"
        artifact_no_link_tree = "tree-3"

        num_iters = 5
        best_iter_1 = 2
        best_iter_2 = 4
        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_1_key,
            artifact_1_tree,
            num_iters,
            best_iter_1,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_2_key,
            artifact_2_tree,
            num_iters,
            best_iter_2,
            ArtifactCategories.dataset,
        )

        # Add non-hyper-param artifact. Single object with iter 0, not pointing at anything
        artifact_body = self._generate_artifact(
            artifact_no_link_key, artifact_no_link_tree
        )
        artifact_body["spec"]["iter"] = 0
        db.store_artifact(db_session, artifact_no_link_key, artifact_body, iter=0)

        results = db.list_artifacts(db_session, name="~artifact")
        # we don't store link artifacts in the DB, so we expect 2 * num_iters - 1, plus a regular artifact
        assert len(results) == (num_iters - 1) * 2 + 1

        results = db.list_artifacts(
            db_session, name=artifact_1_key, best_iteration=True
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        expected_iters = {
            artifact_1_key: best_iter_1,
            artifact_2_key: best_iter_2,
            artifact_no_link_key: 0,
        }
        results = db.list_artifacts(db_session, name="~artifact", best_iteration=True)
        assert len(results) == 3
        for artifact in results:
            artifact_name = artifact["metadata"]["key"]
            assert (
                artifact_name in expected_iters
                and expected_iters[artifact_name] == artifact["spec"]["iter"]
            )

        results = db.list_artifacts(
            db_session, best_iteration=True, category=ArtifactCategories.model
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        # Should get only object-2 (which is of dataset type) without the link artifact
        results = db.list_artifacts(db_session, category=ArtifactCategories.dataset)
        assert len(results) == num_iters - 1
        for artifact in results:
            assert artifact["metadata"]["key"] == artifact_2_key

        # Negative test - asking for both best_iter and iter
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            results = db.list_artifacts(
                db_session, name="~artifact", best_iteration=True, iter=0
            )

    def test_list_artifacts_best_iteration(self, db: DBInterface, db_session: Session):
        artifact_key = "artifact-1"
        artifact_1_tree = "tree-1"
        artifact_2_tree = "tree-2"
        artifact_3_tree = "tree-3"

        num_iters = 5
        best_iter_1 = 2
        best_iter_2 = 4
        best_iter_3 = 1
        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_key,
            artifact_1_tree,
            num_iters,
            best_iter_1,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_key,
            artifact_2_tree,
            num_iters,
            best_iter_2,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
            db,
            db_session,
            artifact_key,
            artifact_3_tree,
            num_iters,
            best_iter_3,
            ArtifactCategories.model,
        )

        for category in [ArtifactCategories.model, None]:
            results = db.list_artifacts(
                db_session, tag="*", best_iteration=True, category=category
            )
            assert len(results) == 3
            for result in results:
                if result["metadata"]["tree"] == artifact_3_tree:
                    assert result["metadata"].get("tag") == "latest"
                else:
                    assert not result["metadata"].get("tag")

    def test_list_artifact_for_tagging_fallback(
        self, db: DBInterface, db_session: Session
    ):
        # create an artifact
        project = "artifact_project"
        artifact_key = "artifact_key_1"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, kind=ArtifactCategories.model
        )
        artifact_tag_1 = "artifact-tag-1"
        db.store_artifact(
            db_session, artifact_key, artifact_body, tag=artifact_tag_1, project=project
        )

        # append artifact tag, but put the `tree` in the `uid` field of the identifier, like older clients do
        identifier = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key,
            uid=artifact_tree,
        )
        artifact_tag_2 = "artifact-tag-2"
        db.append_tag_to_artifacts(db_session, project, artifact_tag_2, [identifier])

        # verify that the artifact has both tags
        artifacts = db.list_artifacts(
            db_session, artifact_key, project=project, tag=artifact_tag_1
        )
        assert len(artifacts) == 1

        artifacts = db.list_artifacts(
            db_session, artifact_key, project=project, tag=artifact_tag_2
        )
        assert len(artifacts) == 1

    def test_iterations_with_latest_tag(self, db: DBInterface, db_session: Session):
        project = "artifact_project"
        artifact_key = "artifact_key"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, project=project
        )
        num_of_iterations = 5

        # create artifacts with the same key and different iterations
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            db.store_artifact(
                db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = db.list_artifacts(db_session, project=project, tag="latest")
        assert len(artifacts) == num_of_iterations

        # mark iteration 3 as the best iteration
        best_iteration = 3
        self._mark_best_iteration_artifact(
            db, db_session, project, artifact_key, artifact_tree, best_iteration
        )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = db.list_artifacts(db_session, project=project, tag="latest")
        assert len(artifacts) == num_of_iterations

        # list artifact with "latest" tag and best_iteration=True - should return only the artifact with iteration 3
        artifacts = db.list_artifacts(
            db_session, project=project, tag="latest", best_iteration=True
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["iter"] == best_iteration

        # run the same test with a different producer id
        artifact_tree_2 = "artifact_tree_2"
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            artifact_body["metadata"]["tree"] = artifact_tree_2
            db.store_artifact(
                db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree_2,
            )

        # list artifact with "latest" tag - should return only the new artifacts
        artifacts = db.list_artifacts(db_session, project=project, tag="latest")
        assert len(artifacts) == num_of_iterations
        producer_ids = set([artifact["metadata"]["tree"] for artifact in artifacts])
        assert len(producer_ids) == 1
        assert producer_ids.pop() == artifact_tree_2

        # mark iteration 2 as the best iteration
        best_iteration = 2
        self._mark_best_iteration_artifact(
            db, db_session, project, artifact_key, artifact_tree_2, best_iteration
        )

        # list artifact with "latest" tag and best iteration - should return only the new artifacts
        artifacts = db.list_artifacts(
            db_session, project=project, tag="latest", best_iteration=True
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["iter"] == best_iteration
        assert artifacts[0]["metadata"]["tree"] == artifact_tree_2

    @pytest.mark.asyncio
    async def test_project_file_counter(self, db: DBInterface, db_session: Session):
        # create artifact with 5 distinct keys, and 3 tags for each key
        project = "artifact_project"
        for i in range(5):
            artifact_key = f"artifact_key_{i}"
            artifact_tree = f"artifact_tree_{i}"
            artifact_body = self._generate_artifact(artifact_key, tree=artifact_tree)
            for j in range(3):
                artifact_tag = f"artifact-tag-{j}"
                db.store_artifact(
                    db_session,
                    artifact_key,
                    artifact_body,
                    tag=artifact_tag,
                    project=project,
                    producer_id=artifact_tree,
                )

        # list artifact with "latest" tag - should return 5 artifacts
        artifacts = db.list_artifacts(db_session, project=project, tag="latest")
        assert len(artifacts) == 5

        # query all artifacts tags, should return 15+5=20 tags
        tags = db.list_artifact_tags(db_session, project=project)
        assert len(tags) == 20

        # files counters should return the most recent artifacts, for each key -> 5 artifacts
        project_to_files_count = db._calculate_files_counters(db_session)
        assert project_to_files_count[project] == 5

    def test_migrate_artifacts_to_v2(self, db: DBInterface, db_session: Session):
        artifact_tree = "tree1"
        artifact_tag = "artifact-tag-1"
        project = "project1"

        self._create_project(db, db_session, project)

        # create an artifact in the old format
        artifact_key_1 = "artifact1"
        artifact_body_1 = self._generate_artifact(
            artifact_key_1, artifact_tree, "artifact", project=project
        )
        artifact_body_1["metadata"]["iter"] = 2
        artifact_body_1["metadata"]["tag"] = artifact_tag
        db.store_artifact_v1(
            db_session,
            artifact_key_1,
            artifact_body_1,
            artifact_tree,
            project=project,
            tag=artifact_tag,
        )

        # create an artifact without an iteration and tag
        artifact_key_2 = "artifact2"
        artifact_body_2 = self._generate_artifact(
            artifact_key_2, artifact_tree, "model", project=project
        )
        db.store_artifact_v1(
            db_session,
            artifact_key_2,
            artifact_body_2,
            artifact_tree,
            project=project,
        )

        # create a legacy artifact in the old format
        legacy_artifact_key = "legacy-dataset-artifact1"
        legacy_artifact_uid = "legacy-uid1"
        legacy_artifact_tag = "legacy-tag-1"
        legacy_artifact = {
            "key": legacy_artifact_key,
            "tag": legacy_artifact_tag,
            "src_path": "/some/other/path",
            "kind": "dataset",
            "tree": legacy_artifact_uid,
            "length": 100,
            "preview": 5,
        }
        db.store_artifact_v1(
            db_session,
            legacy_artifact_key,
            legacy_artifact,
            legacy_artifact_uid,
            project=project,
            tag=legacy_artifact_tag,
        )

        self._run_artifacts_v2_migration(db, db_session)

        # validate the migration succeeded
        query_all = db._query(
            db_session,
            server.api.db.sqldb.models.ArtifactV2,
        )
        new_artifacts = query_all.all()
        assert len(new_artifacts) == 3

        # validate there are 4 tags in total - the specific tag and the latest tag for each artifact
        query_all_tags = db._query(
            db_session,
            new_artifacts[0].Tag,
        )
        new_artifact_tags = query_all_tags.all()
        assert len(new_artifact_tags) == 5

        for expected in [
            {
                "key": artifact_key_1,
                "uid": artifact_tree,
                "project": project,
                "iter": 2,
                "tag": artifact_tag,
            },
            {
                "key": artifact_key_2,
                "uid": artifact_tree,
                "project": project,
                "iter": 0,
                "tag": None,
            },
            {
                "key": legacy_artifact_key,
                "uid": legacy_artifact_uid,
                "project": None,
                "iter": 0,
                "tag": legacy_artifact_tag,
            },
        ]:
            # TODO: remove this query once the v2 db layer methods are implemented. This is just a temporary workaround
            query = db._query(
                db_session,
                server.api.db.sqldb.models.ArtifactV2,
                key=expected["key"],
            )
            artifact = query.one_or_none()
            assert artifact is not None
            assert artifact.key == expected["key"]
            assert artifact.producer_id == expected["uid"]
            assert artifact.project == expected["project"]
            assert artifact.iteration == expected["iter"]

            artifact_dict = artifact.full_object
            assert len(artifact_dict) > 0
            assert artifact_dict["metadata"]["key"] == expected["key"]
            if expected["project"] is not None:
                assert artifact_dict["metadata"]["project"] == expected["project"]
            else:
                assert "project" not in artifact_dict["metadata"]

            # the uid should be the generated uid and not the original one
            assert artifact_dict["metadata"]["uid"] != expected["uid"]

            if expected["tag"] is not None:
                # query the artifact tags and validate the tag exists
                query = db._query(
                    db_session,
                    artifact.Tag,
                    name=expected["tag"],
                )
                tag = query.one_or_none()
                assert tag is not None

            # validate the original artifact was deleted
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                db.read_artifact_v1(
                    db_session, expected["key"], project=expected["project"]
                )

    def test_migrate_many_artifacts_to_v2(self, db: DBInterface, db_session: Session):
        # create 10 artifacts in 10 projects
        for i in range(10):
            project_name = f"project-{i}"
            self._create_project(db, db_session, project_name)
            for j in range(10):
                artifact_key = f"artifact-{j}"
                artifact_uid = f"uid-{j}"
                artifact_tag = f"artifact-tag-{j}"
                artifact_body = self._generate_artifact(
                    artifact_key, artifact_uid, "artifact"
                )
                artifact_body["metadata"]["project"] = project_name
                artifact_body["metadata"]["tag"] = artifact_tag
                db.store_artifact_v1(
                    db_session,
                    artifact_key,
                    artifact_body,
                    artifact_uid,
                    project=project_name,
                    tag=artifact_tag,
                )

        # validate we have 100 artifacts in the old table
        old_artifacts = db._query(
            db_session,
            server.api.db.sqldb.models.Artifact,
        ).all()
        assert len(old_artifacts) == 100

        self._run_artifacts_v2_migration(db, db_session)

        # validate the migration succeeded
        old_artifacts = db._query(
            db_session,
            server.api.db.sqldb.models.Artifact,
        ).all()
        assert len(old_artifacts) == 0

        new_artifacts = db._query(
            db_session,
            server.api.db.sqldb.models.ArtifactV2,
        ).all()
        assert len(new_artifacts) == 100

        # validate there are 200 tags in total - the specific tag and the latest tag for each artifact
        new_artifact_tags = db._query(
            db_session,
            new_artifacts[0].Tag,
        ).all()
        assert len(new_artifact_tags) == 200

        # validate we have 10 distinct projects in the new table
        new_artifact_projects = db_session.execute(
            select([distinct(server.api.db.sqldb.models.ArtifactV2.project)])
        ).fetchall()
        assert len(new_artifact_projects) == 10

    def test_migrate_artifact_v2_tag(self, db: DBInterface, db_session: Session):
        artifact_key = "artifact1"
        artifact_uid = "uid1"
        artifact_tag = "artifact-tag-1"
        project = "project1"

        # create project
        self._create_project(db, db_session, project)

        # create an artifact in the old format
        artifact_body = self._generate_artifact(artifact_key, artifact_uid, "artifact")
        artifact_body["metadata"]["key"] = artifact_key
        artifact_body["metadata"]["iter"] = 2
        artifact_body["metadata"]["project"] = project
        db.store_artifact_v1(
            db_session,
            artifact_key,
            artifact_body,
            artifact_uid,
            project=project,
            tag=artifact_tag,
        )

        query_all = db._query(
            db_session,
            server.api.db.sqldb.models.Artifact,
        )
        old_artifacts = query_all.all()
        assert len(old_artifacts) == 1

        self._run_artifacts_v2_migration(db, db_session)

        # validate the migration succeeded
        query_all = db._query(
            db_session,
            server.api.db.sqldb.models.ArtifactV2,
        )
        new_artifact = query_all.one()

        # validate there are 2 tags in total - the specific tag and the latest
        query_all_tags = db._query(
            db_session,
            new_artifact.Tag,
        )
        new_artifact_tags = query_all_tags.all()
        assert len(new_artifact_tags) == 2

        # list artifacts with the tags
        for tag in [artifact_tag, "latest"]:
            artifacts = db.list_artifacts(db_session, tag=tag, project=project)
            assert len(artifacts) == 1
            assert artifacts[0]["metadata"]["key"] == artifact_key
            assert artifacts[0]["metadata"]["project"] == project
            assert artifacts[0]["metadata"]["uid"] != artifact_uid

    def test_migrate_artifact_v2_persist_db_key_with_iteration(
        self, db: DBInterface, db_session: Session
    ):
        artifact_key = "artifact"
        artifact_tree = "some-tree"
        artifact_tag = "artifact-tag-1"
        project = "project1"
        db_key = "db-key-1"
        iteration = 2

        # create project
        self._create_project(db, db_session, project)

        # create artifacts in the old format
        artifact_body = self._generate_artifact(artifact_key, artifact_tree, "artifact")
        artifact_body["metadata"]["key"] = artifact_key
        artifact_body["metadata"]["iter"] = iteration
        artifact_body["metadata"]["project"] = project
        artifact_body["spec"]["db_key"] = db_key

        # store the artifact with the db_key
        db.store_artifact_v1(
            db_session,
            db_key,
            artifact_body,
            artifact_tree,
            project=project,
            tag=artifact_tag,
            iter=iteration,
        )

        # validate the artifact was stored with the db_key
        key = f"{iteration}-{db_key}"
        artifact = db.read_artifact_v1(db_session, key, project=project)
        assert artifact["metadata"]["key"] == artifact_key

        # migrate the artifacts to v2
        self._run_artifacts_v2_migration(db, db_session)

        # validate the migration succeeded and the db_key was persisted
        query_all = db._query(
            db_session,
            server.api.db.sqldb.models.ArtifactV2,
        )
        new_artifact = query_all.one()
        assert new_artifact.key == db_key
        assert new_artifact.iteration == iteration

    def test_update_model_spec(self, db: DBInterface, db_session: Session):
        artifact_key = "model1"

        # create a model
        model_body = self._generate_artifact(artifact_key, kind="model")
        db.store_artifact(db_session, artifact_key, model_body)
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update the model with spec that should be ignored in UID calc
        model_body["spec"]["parameters"] = {"p1": 5}
        model_body["spec"]["outputs"] = {"o1": 6}
        model_body["spec"]["metrics"] = {"l1": "a"}
        db.store_artifact(db_session, artifact_key, model_body)
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update spec that should not be ignored
        model_body["spec"]["model_file"] = "some/path"
        db.store_artifact(db_session, artifact_key, model_body)
        artifacts = db.list_artifacts(db_session)
        assert len(artifacts) == 2

        tags = [artifact["metadata"].get("tag", None) for artifact in artifacts]
        assert len(tags) == 2
        assert "latest" in tags
        assert None in tags

        for model in artifacts:
            assert model["metadata"]["key"] == artifact_key
            if model["metadata"].get("tag") == "latest":
                assert model["spec"]["model_file"] == "some/path"
            else:
                assert model["spec"].get("model_file") is None

    def _generate_artifact_with_iterations(
        self, db, db_session, key, tree, num_iters, best_iter, kind, project=""
    ):
        # using reversed so the link artifact will be created last, after all the iterations
        # are already stored
        for iter in reversed(range(num_iters)):
            artifact_body = self._generate_artifact(
                key, kind=kind.value if iter != 0 else "link", tree=tree
            )
            if iter == 0:
                artifact_body["spec"]["link_iteration"] = best_iter
            artifact_body["spec"]["iter"] = iter
            db.store_artifact(
                db_session,
                key,
                artifact_body,
                iter=iter,
                project=project,
                producer_id=tree,
            )

    @staticmethod
    def _generate_artifact(
        key, uid=None, kind="artifact", tree=None, project=None, labels=None
    ):
        artifact = {
            "metadata": {"key": key},
            "spec": {"src_path": "/some/path"},
            "kind": kind,
            "status": {"bla": "blabla"},
        }
        if kind:
            artifact["kind"] = kind
        if uid:
            artifact["metadata"]["uid"] = uid
        if tree:
            artifact["metadata"]["tree"] = tree
        if project:
            artifact["metadata"]["project"] = project
        if labels:
            artifact["metadata"]["labels"] = labels

        return artifact

    @staticmethod
    def _mark_best_iteration_artifact(
        db, db_session, project, artifact_key, artifact_tree, best_iteration
    ):
        item = LinkArtifact(
            artifact_key,
            link_iteration=best_iteration,
            link_key=artifact_key,
            link_tree=artifact_tree,
        )
        item.tree = artifact_tree
        item.iter = best_iteration
        db.store_artifact(
            db_session,
            item.db_key,
            item.to_dict(),
            iter=0,
            project=project,
            producer_id=artifact_tree,
        )

    @staticmethod
    def _create_project(db: DBInterface, db_session: Session, project_name):
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=project_name,
            ),
            spec=mlrun.common.schemas.ProjectSpec(description="some-description"),
        )
        db.create_project(db_session, project)

    @staticmethod
    def _run_artifacts_v2_migration(db: DBInterface, db_session: Session):
        with tempfile.TemporaryDirectory() as temp_dir:
            # change the state file path to the temp directory for the test only
            mlrun.mlconf.artifacts.artifact_migration_state_file_path = (
                temp_dir + "/_artifact_migration_state.json"
            )

            # perform the migration
            server.api.initial_data._migrate_artifacts_table_v2(db, db_session)
