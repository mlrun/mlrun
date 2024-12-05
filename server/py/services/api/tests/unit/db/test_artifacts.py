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
import datetime
import tempfile
import unittest.mock

import deepdiff
import pytest
from sqlalchemy import distinct, select

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.lists
import mlrun.utils
from mlrun.artifacts.base import LinkArtifact
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.document import DocumentArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.artifacts.plots import PlotArtifact, PlotlyArtifact
from mlrun.common.schemas.artifact import ArtifactCategories

import framework.db.sqldb.models
import services.api.initial_data
from framework.db.sqldb.models import ArtifactV2
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestArtifacts(TestDatabaseBase):
    def test_list_artifact_name_filter(self):
        artifact_name_1 = "artifact_name_1"
        artifact_name_2 = "artifact_name_2"
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(artifact_name_1, tree=tree)
        artifact_2 = self._generate_artifact(artifact_name_2, tree=tree)

        self._db.store_artifact(
            self._db_session,
            artifact_name_1,
            artifact_1,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name_2,
            artifact_2,
        )
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 2

        artifacts = self._db.list_artifacts(self._db_session, name=artifact_name_1)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = self._db.list_artifacts(self._db_session, name=artifact_name_2)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

        artifacts = self._db.list_artifacts(self._db_session, name="~artifact_name")
        assert len(artifacts) == 2

    def test_list_artifact_iter_parameter(self):
        artifact_name_1 = "artifact_name_1"
        artifact_name_2 = "artifact_name_2"
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(artifact_name_1, tree=tree)
        artifact_2 = self._generate_artifact(artifact_name_2, tree=tree)

        # Use iters with multiple digits, to make sure filtering them via regex works
        test_iters = [0, 5, 9, 42, 219, 2102]
        for iter in test_iters:
            artifact_1["iter"] = artifact_2["iter"] = iter
            self._db.store_artifact(
                self._db_session, artifact_name_1, artifact_1, iter=iter
            )
            self._db.store_artifact(
                self._db_session, artifact_name_2, artifact_2, iter=iter
            )

        # No filter on iter. All are expected
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == len(test_iters) * 2

        # look for the artifact with the "latest" tag - should return all iterations
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_1, tag="latest"
        )
        assert len(artifacts) == len(test_iters)

        # Look for the various iteration numbers. Note that 0 is a special case due to the db structure
        for iter in test_iters:
            artifacts = self._db.list_artifacts(self._db_session, iter=iter)
            assert len(artifacts) == 2
            for artifact in artifacts:
                assert artifact["iter"] == iter

        # Negative test
        artifacts = self._db.list_artifacts(self._db_session, iter=666)
        assert len(artifacts) == 0

        # Iter filter and a name filter, make sure query composition works
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_1, iter=2102
        )
        assert len(artifacts) == 1

    def test_list_artifact_kind_filter(self):
        artifact_name_1 = "artifact_name_1"
        artifact_kind_1 = PlotlyArtifact.kind
        artifact_name_2 = "artifact_name_2"
        artifact_kind_2 = PlotArtifact.kind
        tree = "artifact_tree"
        artifact_1 = self._generate_artifact(
            artifact_name_1, kind=artifact_kind_1, tree=tree
        )
        artifact_2 = self._generate_artifact(
            artifact_name_2, kind=artifact_kind_2, tree=tree
        )

        self._db.store_artifact(
            self._db_session,
            artifact_name_1,
            artifact_1,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name_2,
            artifact_2,
        )
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 2

        artifacts = self._db.list_artifacts(self._db_session, kind=artifact_kind_1)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = self._db.list_artifacts(self._db_session, kind=artifact_kind_2)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

    def test_list_artifact_category_filter(self):
        artifact_name_1 = "artifact_name_1"
        artifact_kind_1 = PlotlyArtifact.kind
        artifact_name_2 = "artifact_name_2"
        artifact_kind_2 = PlotArtifact.kind
        artifact_name_3 = "artifact_name_3"
        artifact_kind_3 = ModelArtifact.kind
        artifact_name_4 = "artifact_name_4"
        artifact_kind_4 = DatasetArtifact.kind
        artifact_name_5 = "artifact_name_5"
        artifact_kind_5 = DocumentArtifact.kind

        artifact_1 = self._generate_artifact(artifact_name_1, kind=artifact_kind_1)
        artifact_2 = self._generate_artifact(artifact_name_2, kind=artifact_kind_2)
        artifact_3 = self._generate_artifact(artifact_name_3, kind=artifact_kind_3)
        artifact_4 = self._generate_artifact(artifact_name_4, kind=artifact_kind_4)
        artifact_5 = self._generate_artifact(artifact_name_5, kind=artifact_kind_5)

        for artifact_name, artifact_object in [
            (artifact_name_1, artifact_1),
            (artifact_name_2, artifact_2),
            (artifact_name_3, artifact_3),
            (artifact_name_4, artifact_4),
            (artifact_name_5, artifact_5),
        ]:
            self._db.store_artifact(
                self._db_session,
                artifact_name,
                artifact_object,
            )

        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 5

        artifacts = self._db.list_artifacts(
            self._db_session, category=mlrun.common.schemas.ArtifactCategories.model
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_3

        artifacts = self._db.list_artifacts(
            self._db_session, category=mlrun.common.schemas.ArtifactCategories.dataset
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_4

        artifacts = self._db.list_artifacts(
            self._db_session, category=mlrun.common.schemas.ArtifactCategories.document
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_5

        artifacts = self._db.list_artifacts(
            self._db_session, category=mlrun.common.schemas.ArtifactCategories.other
        )
        assert len(artifacts) == 2
        assert artifacts[1]["metadata"]["key"] == artifact_name_1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

    def test_list_artifact_label_filter(self):
        total_artifacts = 5
        for i in range(1, total_artifacts + 1):
            artifact_name = f"artifact_name_{i}"
            artifact_tree = f"tree_{i}"
            artifact_labels = {"same_key": "same_value", f"label_{i}": f"value_{i}"}
            artifact = self._generate_artifact(
                artifact_name, tree=artifact_tree, labels=artifact_labels
            )
            self._db.store_artifact(
                self._db_session,
                artifact_name,
                artifact,
            )

        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="same_key=same_value"
        )
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(self._db_session, labels="same_key")
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(self._db_session, labels="~label")
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(self._db_session, labels="~LaBeL=~VALue")
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(self._db_session, labels="label_1=~Value")
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(self._db_session, labels="label_1=value_1")
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(self._db_session, labels="label_1=value_2")
        assert len(artifacts) == 0

        artifacts = self._db.list_artifacts(self._db_session, labels="label_2=~VALUE_2")
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == "artifact_name_2"

    def test_store_artifact_tagging(self):
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_tag = "artifact_tag_1"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_1_kind = PlotlyArtifact.kind
        artifact_1_with_kind_tree = "artifact_tree_2"
        artifact_2_tag = "artifact_tag_2"
        artifact_1_with_kind_body = self._generate_artifact(
            artifact_1_key, kind=artifact_1_kind, tree=artifact_1_with_kind_tree
        )

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_with_kind_body,
            tag=artifact_2_tag,
        )
        artifact = self._db.read_artifact(
            self._db_session, artifact_1_key, tag=artifact_1_tag
        )
        assert artifact["kind"] == "artifact"
        artifact = self._db.read_artifact(
            self._db_session, artifact_1_key, tag="latest", raise_on_not_found=False
        )
        assert artifact is not None
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, tag=artifact_2_tag
        )
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == artifact_1_kind
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, tag="latest"
        )
        assert len(artifacts) == 1

    def test_store_artifact_latest_tag(self):
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

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_2_body,
            project=project,
        )

        artifact_tags = self._db.list_artifact_tags(self._db_session, project)

        # make sure only a single "latest" tag is returned
        assert len(artifact_tags) == 1

        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project
        )
        assert len(artifacts) == 2
        for artifact in artifacts:
            if artifact["metadata"].get("tag") == "latest":
                assert artifact["spec"]["something"] == "different"
            else:
                assert artifact["spec"]["something"] == "same"

    def test_list_artifact_tags_with_category(self):
        project = "artifact_project"
        artifact_1_key, artifact_1_tag = "artifact_key_1", "v1"
        artifact_2_key, artifact_2_tag = "artifact_key_2", "v2"
        artifact_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key,
            tree=artifact_tree,
            project=project,
            kind=mlrun.common.schemas.ArtifactCategories.dataset,
            tag=artifact_1_tag,
        )
        artifact_2_body = self._generate_artifact(
            artifact_2_key,
            tree=artifact_tree,
            project=project,
            kind=mlrun.common.schemas.ArtifactCategories.dataset.model,
            tag=artifact_2_tag,
        )

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=project,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            project=project,
            tag=artifact_2_tag,
        )

        artifact_tags = self._db.list_artifact_tags(self._db_session, project)
        # latest, v1, v2
        assert len(artifact_tags) == 3
        artifact_tags = self._db.list_artifact_tags(
            self._db_session,
            project,
            category=mlrun.common.schemas.ArtifactCategories.dataset,
        )
        assert len(artifact_tags) == 2
        assert artifact_1_tag in artifact_tags
        assert "latest" in artifact_tags
        artifact_tags = self._db.list_artifact_tags(
            self._db_session,
            project,
            category=mlrun.common.schemas.ArtifactCategories.model,
        )
        assert len(artifact_tags) == 2
        assert artifact_2_tag in artifact_tags
        assert "latest" in artifact_tags

    def test_store_artifact_restoring_multiple_tags(self):
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
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            copy.deepcopy(artifact_1_body),
            tag=artifact_1_tag,
            project=project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            copy.deepcopy(artifact_2_body),
            tag=artifact_2_tag,
            project=project,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_key, tag="*", project=project
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
        artifact = self._db.read_artifact(
            self._db_session, artifact_key, tag=artifact_1_tag
        )
        assert artifact["metadata"]["uid"] == expected_uids[0]
        assert artifact["metadata"]["tag"] == artifact_1_tag
        artifact = self._db.read_artifact(
            self._db_session, artifact_key, tag=artifact_2_tag
        )
        assert artifact["metadata"]["uid"] == expected_uids[1]
        assert artifact["metadata"]["tag"] == artifact_2_tag

    def test_store_artifact_with_different_labels(self):
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
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=project,
        )

        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project
        )
        assert len(artifacts) == 1
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["updated"])
        assert mlrun.utils.has_timezone(artifacts[0]["metadata"]["created"])

        # add a new label to the same artifact
        labels["label2"] = "value2"
        artifact_1_body_copy["metadata"]["labels"] = labels
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body_copy,
            project=project,
        )

        # verify that the artifact has both labels and it didn't create a new artifact
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project
        )
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

    def test_store_artifact_replace_tag(self):
        project = "artifact_project"
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=project
        )
        artifact_1_tag = "artifact-tag-1"

        artifact_1_uid = self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
            project=project,
        )

        # verify that the artifact has the tag
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project, tag=artifact_1_tag
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_1_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_1_tree

        # create a new artifact with the same key and tag, but a different tree
        artifact_2_tree = "artifact_tree_2"
        artifact_2_body = self._generate_artifact(
            artifact_1_key, tree=artifact_2_tree, project=project
        )

        artifact_2_uid = self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_2_body,
            tag=artifact_1_tag,
            project=project,
        )

        # verify that only the new artifact has the tag
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project, tag=artifact_1_tag
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_2_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_2_tree

        # verify that the old artifact is still there, but without the tag
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_1_key, project=project
        )
        assert len(artifacts) == 3

    def test_store_artifact_with_different_key(self):
        artifact_key = "artifact_key"
        artifact_different_key = "artifact_different_key"
        artifact_tree = "artifact_tree"

        artifact_body = self._generate_artifact(artifact_key, tree=artifact_tree)
        self._db.store_artifact(
            self._db_session,
            artifact_different_key,
            artifact_body,
        )
        artifact = self._db.read_artifact(self._db_session, artifact_different_key)
        assert artifact
        assert artifact["metadata"]["key"] == artifact_key

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(self._db_session, artifact_key)

    def test_store_artifact_with_invalid_key(self):
        # test storing artifact with invalid key & invalid db_key
        # special character is not allowed in the key
        artifact_invalid_key = "artifact@key"
        artifact_valid_key = "artifact_key"
        artifact_body = self._generate_artifact(artifact_invalid_key)

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            self._db.store_artifact(
                self._db_session,
                artifact_invalid_key,
                artifact_body,
            )

        # store the artifact with invalid db_key
        artifact_invalid_db_key = "artifact#!key"
        artifact_body = self._generate_artifact(artifact_valid_key)
        artifact_body["spec"]["db_key"] = artifact_invalid_db_key
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            self._db.store_artifact(
                self._db_session,
                artifact_valid_key,
                artifact_body,
            )

        # store the artifact with valid db_key which is different than the artifact key
        artifact_body = self._generate_artifact(artifact_valid_key)
        artifact_valid_db_key = "artifact_db_key"
        artifact_body["spec"]["db_key"] = artifact_valid_db_key
        self._db.store_artifact(
            self._db_session,
            artifact_valid_key,
            artifact_body,
        )
        artifact = self._db.read_artifact(self._db_session, artifact_valid_key)
        assert artifact
        assert artifact["metadata"]["key"] == artifact_valid_key
        assert artifact["spec"]["db_key"] == artifact_valid_db_key

    def test_read_artifact_tag_resolution(self):
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

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
        )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(self._db_session, artifact_1_key, tag=artifact_2_tag)
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(self._db_session, artifact_2_key, tag=artifact_1_tag)
        # just verifying it's not raising
        self._db.read_artifact(self._db_session, artifact_1_key, tag=artifact_1_tag)
        self._db.read_artifact(self._db_session, artifact_2_key, tag=artifact_2_tag)
        # check list
        artifacts = self._db.list_artifacts(self._db_session, tag=artifact_1_tag)
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(self._db_session, tag=artifact_2_tag)
        assert len(artifacts) == 1

    def test_overwrite_artifact_with_tag(self):
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

        self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_body,
            tag=artifact_1_tag,
            project=project,
        )
        self._db.store_artifact(
            self._db_session,
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
        self._db.overwrite_artifacts_with_tag(
            self._db_session, project, "new-tag", [identifier_1]
        )

        # verify that only the first artifact is with the new tag now
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="new-tag"
        )
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag=artifact_1_tag
        )
        assert len(artifacts) == 0

        # verify that the second artifact's tag did not change
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag=artifact_2_tag
        )
        assert len(artifacts) == 1

    def test_delete_artifacts_tag_filter(self):
        artifact_1_key = "artifact_key_1"
        artifact_2_key = "artifact_key_2"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_2_tree)
        artifact_1_tag = "artifact-tag-one"
        artifact_2_tag = "artifact-tag-two"

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
        )
        self._db.del_artifacts(self._db_session, tag=artifact_1_tag)
        artifacts = self._db.list_artifacts(self._db_session, tag=artifact_1_tag)
        assert len(artifacts) == 0
        artifacts = self._db.list_artifacts(self._db_session, tag=artifact_2_tag)
        assert len(artifacts) == 1
        self._db.del_artifacts(self._db_session, tag=artifact_2_tag)
        artifacts = self._db.list_artifacts(self._db_session, tag=artifact_2_tag)
        assert len(artifacts) == 0

    def test_delete_artifacts_failure(self):
        artifact_1_key = "artifact_key_1"
        artifact_2_key = "artifact_key_2"
        artifact_1_body = self._generate_artifact(artifact_1_key)
        artifact_2_body = self._generate_artifact(artifact_2_key)
        artifact_1_tag = "artifact-tag-one"
        artifact_2_tag = "artifact-tag-two"

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
        )
        with (
            unittest.mock.patch.object(
                self._db, "_delete_multi_objects", return_value=0
            ),
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            self._db.del_artifacts(self._db_session)
        assert "Failed to delete 2 artifacts" in str(exc.value)

        with (
            unittest.mock.patch.object(
                self._db, "_delete_multi_objects", return_value=1
            ),
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            self._db.del_artifacts(self._db_session)
        assert "Failed to delete 1 artifacts" in str(exc.value)

        artifacts = self._db.list_artifacts(self._db_session, as_records=True)
        assert len(artifacts) == 2
        self._db.del_artifacts(self._db_session)
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 0

    def test_delete_artifacts_with_specific_iteration(self):
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
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # make sure all artifacts were created
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations

        # delete the artifact with iteration 3
        self._db.del_artifact(
            self._db_session, project=project, key=artifact_key, iter=3, tag="latest"
        )

        # make sure the artifact with iteration 3 was deleted
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations - 1

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session, artifact_key, project=project, iter=3
            )

    def test_delete_artifacts_with_specific_uid(self):
        project = "artifact_project"
        artifact_key = "artifact_key"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, project=project
        )
        num_of_iterations = 3

        # create artifacts with the same key and different iterations
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # make sure all artifacts were created
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations

        # take the uid of the first artifact
        uid = artifacts[0]["metadata"]["uid"]

        # delete the artifact with the specific uid
        self._db.del_artifact(
            self._db_session, project=project, key=artifact_key, uid=uid
        )

        # make sure the artifact with the specific uid was deleted
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations - 1

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session, artifact_key, project=project, uid=uid
            )

    def test_delete_artifact_tag_filter(self):
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
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact,
                tag=artifact_tag,
                project=project,
            )

        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_1_key
        )
        # Should return 2 tags ('latest' and artifact_1_tag)
        assert len(artifacts) == 2
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag=artifact_2_tag
        )
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag=artifact_2_tag_2
        )
        assert len(artifacts) == 1

        self._db.del_artifact(
            self._db_session, artifact_1_key, project=project, tag=artifact_1_tag
        )
        artifacts = self._db.list_artifacts(self._db_session, name=artifact_1_key)
        assert len(artifacts) == 0

        # Negative test - wrong tag, no deletions
        self._db.del_artifact(
            self._db_session, artifact_2_key, project=project, tag=artifact_1_tag
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_2_key
        )

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

        tags = self._db.list_artifact_tags(self._db_session, project)
        assert len(tags) == 3

        # Delete the artifact object (should delete all tags of the same artifact object)
        self._db.del_artifact(
            self._db_session, artifact_2_key, tag=artifact_2_tag_2, project=project
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, name=artifact_2_key
        )
        assert len(artifacts) == 0

        # Assert all tags were deleted
        tags = self._db.list_artifact_tags(self._db_session, project)
        assert len(tags) == 0

    def test_list_artifacts_exact_name_match(self):
        artifact_1_key = "pre_artifact_key_suffix"
        artifact_2_key = "pre-artifact-key-suffix"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(artifact_1_key, tree=artifact_1_tree)
        artifact_2_body = self._generate_artifact(artifact_2_key, tree=artifact_2_tree)

        # Store each twice - once with no iter, and once with an iter
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
        )
        artifact_1_body["iter"] = 42
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            iter=42,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
        )
        artifact_2_body["iter"] = 42
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            iter=42,
        )

        def _list_and_assert_count(key, count, iter=None):
            results = self._db.list_artifacts(self._db_session, name=key, iter=iter)
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

    def test_list_artifacts_best_iter_with_tagged_iteration(self):
        artifact_key_1 = "artifact-1"
        artifact_key_2 = "artifact-2"
        artifact_tree_1 = "tree-1"
        artifact_tree_2 = "tree-2"
        num_iters = 3
        best_iter = 2
        project = "project1"
        tag = "mytag1"

        self._generate_artifact_with_iterations(
            artifact_key_1,
            artifact_tree_1,
            num_iters,
            best_iter,
            ArtifactCategories.model,
            project=project,
        )

        self._generate_artifact_with_iterations(
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
        self._db.append_tag_to_artifacts(
            self._db_session, project, tag, [identifier_1, identifier_2]
        )
        results = self._db.list_artifacts(
            self._db_session, project=project, tag=tag, best_iteration=True
        )
        assert len(results) == 2

        for artifact in results:
            assert (
                artifact["metadata"]["tag"] == tag
                and artifact["spec"]["iter"] == best_iter
                and artifact["metadata"]["key"] in (artifact_key_1, artifact_key_2)
            )

    def test_list_artifacts_best_iter(self):
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
            artifact_1_key,
            artifact_1_tree,
            num_iters,
            best_iter_1,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
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
        self._db.store_artifact(
            self._db_session, artifact_no_link_key, artifact_body, iter=0
        )

        results = self._db.list_artifacts(self._db_session, name="~artifact")
        # we don't store link artifacts in the DB, so we expect 2 * num_iters - 1, plus a regular artifact
        assert len(results) == (num_iters - 1) * 2 + 1

        results = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, best_iteration=True
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        expected_iters = {
            artifact_1_key: best_iter_1,
            artifact_2_key: best_iter_2,
            artifact_no_link_key: 0,
        }
        results = self._db.list_artifacts(
            self._db_session, name="~artifact", best_iteration=True
        )
        assert len(results) == 3
        for artifact in results:
            artifact_name = artifact["metadata"]["key"]
            assert (
                artifact_name in expected_iters
                and expected_iters[artifact_name] == artifact["spec"]["iter"]
            )

        results = self._db.list_artifacts(
            self._db_session, best_iteration=True, category=ArtifactCategories.model
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        # Should get only object-2 (which is of dataset type) without the link artifact
        results = self._db.list_artifacts(
            self._db_session, category=ArtifactCategories.dataset
        )
        assert len(results) == num_iters - 1
        for artifact in results:
            assert artifact["metadata"]["key"] == artifact_2_key

        # Negative test - asking for both best_iter and iter
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            results = self._db.list_artifacts(
                self._db_session, name="~artifact", best_iteration=True, iter=0
            )

    def test_list_artifacts_best_iteration(self):
        artifact_key = "artifact-1"
        artifact_1_tree = "tree-1"
        artifact_2_tree = "tree-2"
        artifact_3_tree = "tree-3"

        num_iters = 5
        best_iter_1 = 2
        best_iter_2 = 4
        best_iter_3 = 1
        self._generate_artifact_with_iterations(
            artifact_key,
            artifact_1_tree,
            num_iters,
            best_iter_1,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
            artifact_key,
            artifact_2_tree,
            num_iters,
            best_iter_2,
            ArtifactCategories.model,
        )
        self._generate_artifact_with_iterations(
            artifact_key,
            artifact_3_tree,
            num_iters,
            best_iter_3,
            ArtifactCategories.model,
        )

        for category in [ArtifactCategories.model, None]:
            results = self._db.list_artifacts(
                self._db_session, tag="*", best_iteration=True, category=category
            )
            assert len(results) == 3
            for result in results:
                if result["metadata"]["tree"] == artifact_3_tree:
                    assert result["metadata"].get("tag") == "latest"
                else:
                    assert not result["metadata"].get("tag")

    def test_list_artifact_for_tagging_fallback(self):
        # create an artifact
        project = "artifact_project"
        artifact_key = "artifact_key_1"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, kind=ArtifactCategories.model
        )
        artifact_tag_1 = "artifact-tag-1"
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_body,
            tag=artifact_tag_1,
            project=project,
        )

        # append artifact tag, but put the `tree` in the `uid` field of the identifier, like older clients do
        identifier = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key,
            uid=artifact_tree,
        )
        artifact_tag_2 = "artifact-tag-2"
        self._db.append_tag_to_artifacts(
            self._db_session, project, artifact_tag_2, [identifier]
        )

        # verify that the artifact has both tags
        artifacts = self._db.list_artifacts(
            self._db_session, artifact_key, project=project, tag=artifact_tag_1
        )
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(
            self._db_session, artifact_key, project=project, tag=artifact_tag_2
        )
        assert len(artifacts) == 1

    @pytest.mark.parametrize("limit", [None, 6])
    def test_list_artifacts_returns_elements_by_order_updated_field(self, limit):
        project = "artifact_project"

        # Create artifacts
        number_of_artifacts = 10
        for counter in range(number_of_artifacts):
            artifact_key = f"artifact-{counter}"
            artifact_body = self._generate_artifact(artifact_key, project=project)
            self._db.store_artifact(
                self._db_session, artifact_key, artifact_body, project=project
            )

        artifacts = self._db.list_artifacts(
            self._db_session, project=project, limit=limit
        )

        expected_count = limit or number_of_artifacts
        assert (
            len(artifacts) == expected_count
        ), f"Expected {expected_count} results, got {len(artifacts)}"

        start_index = number_of_artifacts - 1
        expected_names = [
            f"artifact-{i}"
            for i in range(start_index, start_index - expected_count, -1)
        ]

        for artifact, expected_name in zip(artifacts, expected_names):
            artifact_name = artifact["metadata"]["key"]
            assert (
                artifact_name == expected_name
            ), f"Expected {expected_name}, got {artifact_name}"

    def test_list_artifacts_producer_uri(self):
        project = "artifact_project"
        artifact_key = "dummy-artifact"

        def store_artifact_with_producer(artifact_key, project, producer_uri, tag):
            producer = {"uri": producer_uri}
            artifact_body = self._generate_artifact(
                artifact_key, project=project, producer=producer, tag=tag
            )
            self._db.store_artifact(
                self._db_session, artifact_key, artifact_body, project=project
            )

        producer_uri_without_iteration = f"{project}/dummy-run-id"
        first_producer_uri = f"{producer_uri_without_iteration}-0"
        store_artifact_with_producer(
            artifact_key, project, first_producer_uri, tag="v1"
        )

        second_producer_uri = f"{producer_uri_without_iteration}-1"
        store_artifact_with_producer(
            artifact_key, project, second_producer_uri, tag="v2"
        )

        artifacts = self._db.list_artifacts(
            self._db_session,
            project=project,
            producer_uri=producer_uri_without_iteration,
        )

        assert len(artifacts) == 2, f"Expected 2 artifacts, but found {len(artifacts)}"
        assert (
            artifacts[0]["spec"]["producer"]["uri"] == second_producer_uri
        ), f"Expected producer URI {second_producer_uri}, but got {artifacts[0]['spec']['producer']['uri']}"

        assert (
            artifacts[1]["spec"]["producer"]["uri"] == first_producer_uri
        ), f"Expected producer URI {first_producer_uri}, but got {artifacts[1]['spec']['producer']['uri']}"

    def test_iterations_with_latest_tag(self):
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
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations

        # mark iteration 3 as the best iteration
        best_iteration = 3
        self._mark_best_iteration_artifact(
            project, artifact_key, artifact_tree, best_iteration
        )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations

        # list artifact with "latest" tag and best_iteration=True - should return only the artifact with iteration 3
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest", best_iteration=True
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["iter"] == best_iteration

        # run the same test with a different producer id
        artifact_tree_2 = "artifact_tree_2"
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            artifact_body["metadata"]["tree"] = artifact_tree_2
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=project,
                iter=iteration,
                producer_id=artifact_tree_2,
            )

        # list artifact with "latest" tag - should return only the new artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations
        producer_ids = set([artifact["metadata"]["tree"] for artifact in artifacts])
        assert len(producer_ids) == 1
        assert producer_ids.pop() == artifact_tree_2

        # mark iteration 2 as the best iteration
        best_iteration = 2
        self._mark_best_iteration_artifact(
            project, artifact_key, artifact_tree_2, best_iteration
        )

        # list artifact with "latest" tag and best iteration - should return only the new artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest", best_iteration=True
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["iter"] == best_iteration
        assert artifacts[0]["metadata"]["tree"] == artifact_tree_2

    @pytest.mark.asyncio
    async def test_project_file_counter(self):
        # create artifact with 5 distinct keys, and 3 tags for each key
        project = "artifact_project"
        for i in range(5):
            artifact_key = f"artifact_key_{i}"
            artifact_tree = f"artifact_tree_{i}"
            artifact_body = self._generate_artifact(artifact_key, tree=artifact_tree)
            for j in range(3):
                artifact_tag = f"artifact-tag-{j}"
                self._db.store_artifact(
                    self._db_session,
                    artifact_key,
                    artifact_body,
                    tag=artifact_tag,
                    project=project,
                    producer_id=artifact_tree,
                )

        # list artifact with "latest" tag - should return 5 artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=project, tag="latest"
        )
        assert len(artifacts) == 5

        # query all artifacts tags, should return 4 tags = 3 tags + latest
        tags = self._db.list_artifact_tags(self._db_session, project=project)
        assert len(tags) == 4

        # files counters should return the most recent artifacts, for each key -> 5 artifacts
        project_to_files_count = self._db._calculate_files_counters(self._db_session)
        assert project_to_files_count[project] == 5

    def test_migrate_artifacts_to_v2(self):
        artifact_tree = "tree1"
        artifact_tag = "artifact-tag-1"
        project = "project1"

        self._create_project(project)

        # create an artifact in the old format
        artifact_key_1 = "artifact1"
        artifact_body_1 = self._generate_artifact(
            artifact_key_1, artifact_tree, "artifact", project=project
        )
        artifact_body_1["metadata"]["iter"] = 2
        artifact_body_1["metadata"]["tag"] = artifact_tag
        self._db.store_artifact_v1(
            self._db_session,
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
        self._db.store_artifact_v1(
            self._db_session,
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
        self._db.store_artifact_v1(
            self._db_session,
            legacy_artifact_key,
            legacy_artifact,
            legacy_artifact_uid,
            project=project,
            tag=legacy_artifact_tag,
        )

        self._run_artifacts_v2_migration()

        # validate the migration succeeded
        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        )
        new_artifacts = query_all.all()
        assert len(new_artifacts) == 3

        # validate there are 4 tags in total - the specific tag and the latest tag for each artifact
        query_all_tags = self._db._query(
            self._db_session,
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
            query = self._db._query(
                self._db_session,
                framework.db.sqldb.models.ArtifactV2,
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
                query = self._db._query(
                    self._db_session,
                    artifact.Tag,
                    name=expected["tag"],
                )
                tag = query.one_or_none()
                assert tag is not None

            # validate the original artifact was deleted
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self._db.read_artifact_v1(
                    self._db_session, expected["key"], project=expected["project"]
                )

    def test_migrate_many_artifacts_to_v2(self):
        # create 10 artifacts in 10 projects
        for i in range(10):
            project_name = f"project-{i}"
            self._create_project(project_name)
            for j in range(10):
                artifact_key = f"artifact-{j}"
                artifact_uid = f"uid-{j}"
                artifact_tag = f"artifact-tag-{j}"
                artifact_body = self._generate_artifact(
                    artifact_key, artifact_uid, "artifact"
                )
                artifact_body["metadata"]["project"] = project_name
                artifact_body["metadata"]["tag"] = artifact_tag
                self._db.store_artifact_v1(
                    self._db_session,
                    artifact_key,
                    artifact_body,
                    artifact_uid,
                    project=project_name,
                    tag=artifact_tag,
                )

        # validate we have 100 artifacts in the old table
        old_artifacts = self._db._query(
            self._db_session,
            framework.db.sqldb.models.Artifact,
        ).all()
        assert len(old_artifacts) == 100

        self._run_artifacts_v2_migration()

        # validate the migration succeeded
        old_artifacts = self._db._query(
            self._db_session,
            framework.db.sqldb.models.Artifact,
        ).all()
        assert len(old_artifacts) == 0

        new_artifacts = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        ).all()
        assert len(new_artifacts) == 100

        # validate there are 200 tags in total - the specific tag and the latest tag for each artifact
        new_artifact_tags = self._db._query(
            self._db_session,
            new_artifacts[0].Tag,
        ).all()
        assert len(new_artifact_tags) == 200

        # validate we have 10 distinct projects in the new table
        new_artifact_projects = self._db_session.execute(
            select([distinct(framework.db.sqldb.models.ArtifactV2.project)])
        ).fetchall()
        assert len(new_artifact_projects) == 10

    def test_migrate_artifact_v2_tag(self):
        artifact_key = "artifact1"
        artifact_uid = "uid1"
        artifact_tag = "artifact-tag-1"
        project = "project1"

        # create project
        self._create_project(project)

        # create an artifact in the old format
        artifact_body = self._generate_artifact(artifact_key, artifact_uid, "artifact")
        artifact_body["metadata"]["key"] = artifact_key
        artifact_body["metadata"]["iter"] = 2
        artifact_body["metadata"]["project"] = project
        self._db.store_artifact_v1(
            self._db_session,
            artifact_key,
            artifact_body,
            artifact_uid,
            project=project,
            tag=artifact_tag,
        )

        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.Artifact,
        )
        old_artifacts = query_all.all()
        assert len(old_artifacts) == 1

        self._run_artifacts_v2_migration()

        # validate the migration succeeded
        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        )
        new_artifact = query_all.one()

        # validate there are 2 tags in total - the specific tag and the latest
        query_all_tags = self._db._query(
            self._db_session,
            new_artifact.Tag,
        )
        new_artifact_tags = query_all_tags.all()
        assert len(new_artifact_tags) == 2

        # list artifacts with the tags
        for tag in [artifact_tag, "latest"]:
            artifacts = self._db.list_artifacts(
                self._db_session, tag=tag, project=project
            )
            assert len(artifacts) == 1
            assert artifacts[0]["metadata"]["key"] == artifact_key
            assert artifacts[0]["metadata"]["project"] == project
            assert artifacts[0]["metadata"]["uid"] != artifact_uid

    def test_migrate_artifact_v2_persist_db_key_with_iteration(self):
        artifact_key = "artifact"
        artifact_tree = "some-tree"
        artifact_tag = "artifact-tag-1"
        project = "project1"
        db_key = "db-key-1"
        iteration = 2

        # create project
        self._create_project(project)

        # create artifacts in the old format
        artifact_body = self._generate_artifact(artifact_key, artifact_tree, "artifact")
        artifact_body["metadata"]["key"] = artifact_key
        artifact_body["metadata"]["iter"] = iteration
        artifact_body["metadata"]["project"] = project
        artifact_body["spec"]["db_key"] = db_key

        # store the artifact with the db_key
        self._db.store_artifact_v1(
            self._db_session,
            db_key,
            artifact_body,
            artifact_tree,
            project=project,
            tag=artifact_tag,
            iter=iteration,
        )

        # validate the artifact was stored with the db_key
        key = f"{iteration}-{db_key}"
        artifact = self._db.read_artifact_v1(self._db_session, key, project=project)
        assert artifact["metadata"]["key"] == artifact_key

        # migrate the artifacts to v2
        self._run_artifacts_v2_migration()

        # validate the migration succeeded and the db_key was persisted
        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        )
        new_artifact = query_all.one()
        assert new_artifact.key == db_key
        assert new_artifact.iteration == iteration

    def test_migrate_artifact_without_metadata_key(self):
        # empty key on purpose
        artifact_key = ""
        artifact_tree = "some-tree"
        artifact_tag = "artifact-tag-1"
        project = "project1"
        db_key = "db-key-1"

        # create project
        self._create_project(project)

        # create artifacts in the old format
        artifact_body = self._generate_artifact(artifact_key, artifact_tree, "artifact")
        artifact_body["metadata"]["project"] = project
        artifact_body["spec"]["db_key"] = db_key

        # store the artifact with the db_key
        self._db.store_artifact_v1(
            self._db_session,
            db_key,
            artifact_body,
            artifact_tree,
            project=project,
            tag=artifact_tag,
        )

        # validate the artifact was stored with the db_key
        artifact = self._db.read_artifact_v1(self._db_session, db_key, project=project)
        assert artifact["metadata"]["key"] == ""
        assert artifact["spec"]["db_key"] == db_key

        # migrate the artifacts to v2
        self._run_artifacts_v2_migration()

        # validate the migration succeeded and the metadata key is the db_key
        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        )
        new_artifact = query_all.one()

        assert new_artifact.key == db_key
        artifact = new_artifact.full_object
        assert artifact["metadata"]["key"] == db_key

    def test_migrate_invalid_artifact(self):
        # create an artifact with an invalid struct
        artifact = framework.db.sqldb.models.Artifact(
            project="my-project",
            key="my-key",
            updated=datetime.datetime.now(),
            uid="something",
        )
        artifact.struct = {"something": "blabla"}

        self._db_session.add(artifact)
        self._db_session.commit()

        self._run_artifacts_v2_migration()

        query_all = self._db._query(
            self._db_session,
            framework.db.sqldb.models.ArtifactV2,
        )
        new_artifacts = query_all.all()

        assert len(new_artifacts) == 1

    def test_update_model_spec(self):
        artifact_key = "model1"

        # create a model
        model_body = self._generate_artifact(artifact_key, kind="model")
        self._db.store_artifact(self._db_session, artifact_key, model_body)
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update the model with spec that should be ignored in UID calc
        model_body["spec"]["parameters"] = {"p1": 5}
        model_body["spec"]["outputs"] = {"o1": 6}
        model_body["spec"]["metrics"] = {"l1": "a"}
        self._db.store_artifact(self._db_session, artifact_key, model_body)
        artifacts = self._db.list_artifacts(self._db_session)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update spec that should not be ignored
        model_body["spec"]["model_file"] = "some/path"
        self._db.store_artifact(self._db_session, artifact_key, model_body)
        artifacts = self._db.list_artifacts(self._db_session)
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

    def test_read_and_list_artifacts_with_tags(self):
        k1, t1, art1 = "k1", "t1", {"a": 1, "b": "blubla"}
        t2, art2 = "t2", {"a": 2, "b": "blublu"}
        prj = "p38"
        self._db.store_artifact(
            self._db_session, k1, art1, producer_id=t1, iter=1, project=prj, tag="tag1"
        )
        self._db.store_artifact(
            self._db_session, k1, art2, producer_id=t2, iter=2, project=prj, tag="tag2"
        )

        result = self._db.read_artifact(
            self._db_session, k1, "tag1", iter=1, project=prj
        )
        assert result["metadata"]["tag"] == "tag1"
        result = self._db.read_artifact(
            self._db_session, k1, "tag2", iter=2, project=prj
        )
        assert result["metadata"]["tag"] == "tag2"
        result = self._db.read_artifact(self._db_session, k1, iter=1, project=prj)
        # When doing get without a tag, the returned object must not contain a tag.
        assert "tag" not in result["metadata"]

        result = self._db.list_artifacts(self._db_session, k1, project=prj, tag="*")
        assert len(result) == 3
        for artifact in result:
            assert (
                (artifact["a"] == 1 and artifact["metadata"]["tag"] == "tag1")
                or (artifact["a"] == 2 and artifact["metadata"]["tag"] == "tag2")
                or (artifact["a"] in (1, 2) and artifact["metadata"]["tag"] == "latest")
            )

        # To be used later, after adding tags
        full_results = result

        result = self._db.list_artifacts(self._db_session, k1, tag="tag1", project=prj)
        assert (
            len(result) == 1
            and result[0]["metadata"]["tag"] == "tag1"
            and result[0]["a"] == 1
        )
        result = self._db.list_artifacts(self._db_session, k1, tag="tag2", project=prj)
        assert (
            len(result) == 1
            and result[0]["metadata"]["tag"] == "tag2"
            and result[0]["a"] == 2
        )

        # Add another tag to all objects (there are 2 at this point)
        new_tag = "new-tag"
        expected_results = mlrun.lists.ArtifactList()
        for artifact in full_results:
            expected_results.append(artifact)
            if artifact["metadata"]["tag"] == "latest":
                # We don't want to add a new tag to the "latest" object (it's the same object as the one with
                # tag "tag2")
                continue
            artifact_with_new_tag = copy.deepcopy(artifact)
            artifact_with_new_tag["metadata"]["tag"] = new_tag
            expected_results.append(artifact_with_new_tag)

        artifacts = self._db_session.query(ArtifactV2).all()
        self._db.tag_objects_v2(
            self._db_session, artifacts, prj, name=new_tag, obj_name_attribute="key"
        )
        result = self._db.list_artifacts(self._db_session, k1, prj, tag="*")
        assert deepdiff.DeepDiff(result, expected_results, ignore_order=True) == {}

        # Add another tag to the art1
        self._db.store_artifact(
            self._db_session, k1, art1, producer_id=t1, iter=1, project=prj, tag="tag3"
        )
        # this makes it the latest object of this key, so we need to remove the artifact
        # with tag "latest" from the expected results
        expected_results = mlrun.lists.ArtifactList(
            [
                artifact
                for artifact in expected_results
                if artifact["metadata"]["tag"] != "latest"
            ]
        )

        result = self._db.read_artifact(
            self._db_session, k1, "tag3", iter=1, project=prj
        )
        assert result["metadata"]["tag"] == "tag3"
        expected_results.append(copy.deepcopy(result))

        # add it again but with the "latest" tag
        result["metadata"]["tag"] = "latest"
        expected_results.append(result)

        result = self._db.list_artifacts(self._db_session, k1, prj, tag="*")
        # We want to ignore the "updated" field, since it changes as we store a new tag.
        exclude_regex = r"root\[\d+\]\['updated'\]"
        assert (
            deepdiff.DeepDiff(
                result,
                expected_results,
                ignore_order=True,
                exclude_regex_paths=exclude_regex,
            )
            == {}
        )

    def test_artifacts_latest(self):
        k1, t1, art1 = "k1", "t1", {"a": 1}
        prj = "p38"
        self._db.store_artifact(self._db_session, k1, art1, producer_id=t1, project=prj)

        arts = self._db.list_artifacts(self._db_session, project=prj, tag="latest")
        assert art1["a"] == arts[0]["a"], "bad artifact"

        t2, art2 = "t2", {"a": 17}
        self._db.store_artifact(self._db_session, k1, art2, producer_id=t2, project=prj)
        arts = self._db.list_artifacts(self._db_session, project=prj, tag="latest")
        assert 1 == len(arts), "count"
        assert art2["a"] == arts[0]["a"], "bad artifact"

        k2, t3, art3 = "k2", "t3", {"a": 99}
        self._db.store_artifact(self._db_session, k2, art3, producer_id=t3, project=prj)
        arts = self._db.list_artifacts(self._db_session, project=prj, tag="latest")
        assert 2 == len(arts), "number"
        assert {17, 99} == set(art["a"] for art in arts), "latest"

    def test_list_artifact_tags(self):
        self._db.store_artifact(
            self._db_session, "k1", {}, producer_id="1", tag="t1", project="p1"
        )
        self._db.store_artifact(
            self._db_session, "k1", {}, producer_id="2", tag="t2", project="p1"
        )
        self._db.store_artifact(
            self._db_session, "k1", {}, producer_id="2", tag="t2", project="p2"
        )
        self._db.store_artifact(
            self._db_session,
            "k2",
            {"kind": ModelArtifact.kind},
            producer_id="3",
            tag="t3",
            project="p1",
        )
        self._db.store_artifact(
            self._db_session,
            "k3",
            {"kind": DatasetArtifact.kind},
            producer_id="4",
            tag="t4",
            project="p2",
        )

        self._db.store_artifact(
            self._db_session,
            "k4",
            {"kind": DocumentArtifact.kind},
            producer_id="5",
            tag="t5",
            project="p2",
        )

        tags = self._db.list_artifact_tags(self._db_session, "p1")
        expected_tags = [
            "t1",
            "latest",
            "t2",
            "t3",
        ]
        assert deepdiff.DeepDiff(tags, expected_tags, ignore_order=True) == {}

        tags = self._db.list_artifact_tags(self._db_session, "p2")
        expected_tags = [
            "t2",
            "t4",
            "latest",
            "t5",
        ]
        assert deepdiff.DeepDiff(tags, expected_tags, ignore_order=True) == {}

        # filter by category
        model_tags = self._db.list_artifact_tags(
            self._db_session, "p1", mlrun.common.schemas.ArtifactCategories.model
        )
        expected_tags = ["t3", "latest"]
        assert deepdiff.DeepDiff(expected_tags, model_tags, ignore_order=True) == {}

        dataset_tags = self._db.list_artifact_tags(
            self._db_session, "p2", mlrun.common.schemas.ArtifactCategories.dataset
        )
        expected_tags = ["t4", "latest"]
        assert deepdiff.DeepDiff(expected_tags, dataset_tags, ignore_order=True) == {}

        document_tags = self._db.list_artifact_tags(
            self._db_session, "p2", mlrun.common.schemas.ArtifactCategories.document
        )
        expected_tags = ["t5", "latest"]
        assert deepdiff.DeepDiff(expected_tags, document_tags, ignore_order=True) == {}

    def test_list_artifact_date(self):
        t1 = datetime.datetime(2020, 2, 16)
        t2 = t1 - datetime.timedelta(days=7)
        t3 = t2 - datetime.timedelta(days=7)
        project = "p7"

        # create artifacts in the db directly to avoid the store_artifact function which sets the updated field
        artifacts_to_create = []
        for key, updated, producer_id in [
            ("k1", t1, "p1"),
            ("k2", t2, "p2"),
            ("k3", t3, "p3"),
        ]:
            artifact_struct = mlrun.artifacts.Artifact(
                metadata=mlrun.artifacts.ArtifactMetadata(
                    key=key, project=project, tree=producer_id
                ),
                spec=mlrun.artifacts.ArtifactSpec(),
            )
            db_artifact = ArtifactV2(
                project=project, key=key, updated=updated, producer_id=producer_id
            )
            db_artifact.full_object = artifact_struct.to_dict()
            artifacts_to_create.append(db_artifact)

        self._db._upsert(self._db_session, artifacts_to_create)

        arts = self._db.list_artifacts(
            self._db_session, project=project, since=t3, tag="*"
        )
        assert 3 == len(arts), "since t3"

        arts = self._db.list_artifacts(
            self._db_session, project=project, since=t2, tag="*"
        )
        assert 2 == len(arts), "since t2"

        arts = self._db.list_artifacts(
            self._db_session,
            project=project,
            since=t1 + datetime.timedelta(days=1),
            tag="*",
        )
        assert not arts, "since t1+"

        arts = self._db.list_artifacts(
            self._db_session, project=project, until=t2, tag="*"
        )
        assert 2 == len(arts), "until t2"

        arts = self._db.list_artifacts(
            self._db_session, project=project, since=t2, until=t2, tag="*"
        )
        assert 1 == len(arts), "since/until t2"

    def _generate_artifact_with_iterations(
        self, key, tree, num_iters, best_iter, kind, project=""
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
            self._db.store_artifact(
                self._db_session,
                key,
                artifact_body,
                iter=iter,
                project=project,
                producer_id=tree,
            )

    @staticmethod
    def _generate_artifact(
        key,
        uid=None,
        kind="artifact",
        tree=None,
        project=None,
        labels=None,
        tag=None,
        producer=None,
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
        if tag:
            artifact["metadata"]["tag"] = tag
        if producer:
            artifact["spec"]["producer"] = producer

        return artifact

    def _mark_best_iteration_artifact(
        self, project, artifact_key, artifact_tree, best_iteration
    ):
        item = LinkArtifact(
            artifact_key,
            link_iteration=best_iteration,
            link_key=artifact_key,
            link_tree=artifact_tree,
        )
        item.tree = artifact_tree
        item.iter = best_iteration
        self._db.store_artifact(
            self._db_session,
            item.db_key,
            item.to_dict(),
            iter=0,
            project=project,
            producer_id=artifact_tree,
        )

    def _create_project(self, project_name):
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=project_name,
            ),
            spec=mlrun.common.schemas.ProjectSpec(description="some-description"),
        )
        self._db.create_project(self._db_session, project)

    def _run_artifacts_v2_migration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # change the state file path to the temp directory for the test only
            mlrun.mlconf.artifacts.artifact_migration_state_file_path = (
                temp_dir + "/_artifact_migration_state.json"
            )

            # perform the migration
            services.api.initial_data._migrate_artifacts_table_v2(
                self._db, self._db_session
            )
