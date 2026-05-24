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

import copy
import datetime
import unittest.mock

import deepdiff
import pytest
from sqlalchemy.orm import Query

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.lists
import mlrun.utils
from mlrun.artifacts import Artifact
from mlrun.artifacts.base import LinkArtifact
from mlrun.artifacts.code import CodeArtifact
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.document import DocumentArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.artifacts.plots import PlotArtifact, PlotlyArtifact
from mlrun.common.schemas.artifact import ArtifactCategories

import framework.db.sqldb.models
from framework.db.sqldb.db import SQLDB
from framework.db.sqldb.models import ArtifactV2
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestArtifacts(TestDatabaseBase):
    project = "artifact-project"

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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name_2,
            artifact_2,
            project=self.project,
        )
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 2

        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_1, project=self.project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_2, project=self.project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_2

        artifacts = self._db.list_artifacts(
            self._db_session, name="~artifact_name", project=self.project
        )
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
                self._db_session,
                artifact_name_1,
                artifact_1,
                iter=iter,
                project=self.project,
            )
            self._db.store_artifact(
                self._db_session,
                artifact_name_2,
                artifact_2,
                iter=iter,
                project=self.project,
            )

        # No filter on iter. All are expected
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == len(test_iters) * 2

        # look for the artifact with the "latest" tag - should return all iterations
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_1, tag="latest", project=self.project
        )
        assert len(artifacts) == len(test_iters)

        # Look for the various iteration numbers. Note that 0 is a special case due to the db structure
        for iter in test_iters:
            artifacts = self._db.list_artifacts(
                self._db_session, iter=iter, project=self.project
            )
            assert len(artifacts) == 2
            for artifact in artifacts:
                assert artifact["iter"] == iter

        # Negative test
        artifacts = self._db.list_artifacts(
            self._db_session, iter=666, project=self.project
        )
        assert len(artifacts) == 0

        # Iter filter and a name filter, make sure query composition works
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_name_1, iter=2102, project=self.project
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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name_2,
            artifact_2,
            project=self.project,
        )
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 2

        artifacts = self._db.list_artifacts(
            self._db_session, kind=artifact_kind_1, project=self.project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_1

        artifacts = self._db.list_artifacts(
            self._db_session, kind=artifact_kind_2, project=self.project
        )
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
        artifact_name_6 = "artifact_name_6"
        artifact_kind_6 = CodeArtifact.kind

        artifact_1 = self._generate_artifact(artifact_name_1, kind=artifact_kind_1)
        artifact_2 = self._generate_artifact(artifact_name_2, kind=artifact_kind_2)
        artifact_3 = self._generate_artifact(artifact_name_3, kind=artifact_kind_3)
        artifact_4 = self._generate_artifact(artifact_name_4, kind=artifact_kind_4)
        artifact_5 = self._generate_artifact(artifact_name_5, kind=artifact_kind_5)
        artifact_6 = self._generate_artifact(artifact_name_6, kind=artifact_kind_6)

        for artifact_name, artifact_object in [
            (artifact_name_1, artifact_1),
            (artifact_name_2, artifact_2),
            (artifact_name_3, artifact_3),
            (artifact_name_4, artifact_4),
            (artifact_name_5, artifact_5),
            (artifact_name_6, artifact_6),
        ]:
            self._db.store_artifact(
                self._db_session,
                artifact_name,
                artifact_object,
                project=self.project,
            )

        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 6

        artifacts = self._db.list_artifacts(
            self._db_session,
            category=mlrun.common.schemas.ArtifactCategories.model,
            project=self.project,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_3

        artifacts = self._db.list_artifacts(
            self._db_session,
            category=mlrun.common.schemas.ArtifactCategories.dataset,
            project=self.project,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_4

        artifacts = self._db.list_artifacts(
            self._db_session,
            category=mlrun.common.schemas.ArtifactCategories.document,
            project=self.project,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_5

        artifacts = self._db.list_artifacts(
            self._db_session,
            category=mlrun.common.schemas.ArtifactCategories.other,
            project=self.project,
        )
        assert len(artifacts) == 3
        assert artifacts[0]["metadata"]["key"] == artifact_name_6
        assert artifacts[1]["metadata"]["key"] == artifact_name_2
        assert artifacts[2]["metadata"]["key"] == artifact_name_1

        artifacts = self._db.list_artifacts(
            self._db_session, kind=CodeArtifact.kind, project=self.project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_name_6

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
                project=self.project,
            )

        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="same_key=same_value", project=self.project
        )
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="same_key", project=self.project
        )
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="~label", project=self.project
        )
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="~LaBeL=~VALue", project=self.project
        )
        assert len(artifacts) == total_artifacts

        artifacts = self._db.list_artifacts(
            self._db_session, labels="label_1=~Value", project=self.project
        )
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(
            self._db_session, labels="label_1=value_1", project=self.project
        )
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(
            self._db_session, labels="label_1=value_2", project=self.project
        )
        assert len(artifacts) == 0

        artifacts = self._db.list_artifacts(
            self._db_session, labels="label_2=~VALUE_2", project=self.project
        )
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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_with_kind_body,
            tag=artifact_2_tag,
            project=self.project,
        )
        artifact = self._db.read_artifact(
            self._db_session,
            key=artifact_1_key,
            tag=artifact_1_tag,
            project=self.project,
        )
        assert artifact["kind"] == "artifact"
        artifact = self._db.read_artifact(
            self._db_session,
            key=artifact_1_key,
            tag="latest",
            raise_on_not_found=False,
            project=self.project,
        )
        assert artifact is not None
        artifacts = self._db.list_artifacts(
            self._db_session,
            name=artifact_1_key,
            tag=artifact_2_tag,
            project=self.project,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == artifact_1_kind
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, tag="latest", project=self.project
        )
        assert len(artifacts) == 1

    def test_store_artifact_latest_tag(self):
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=self.project
        )
        artifact_2_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=self.project
        )
        artifact_1_body["spec"]["something"] = "same"
        artifact_2_body["spec"]["something"] = "different"

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_2_body,
            project=self.project,
        )

        artifact_tags = self._db.list_artifact_tags(self._db_session, self.project)

        # make sure only a single "latest" tag is returned
        assert len(artifact_tags) == 1

        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, project=self.project
        )
        assert len(artifacts) == 2
        for artifact in artifacts:
            if artifact["metadata"].get("tag") == "latest":
                assert artifact["spec"]["something"] == "different"
            else:
                assert artifact["spec"]["something"] == "same"

    def test_list_artifact_tags_with_category(self):
        artifact_1_key, artifact_1_tag = "artifact_key_1", "v1"
        artifact_2_key, artifact_2_tag = "artifact_key_2", "v2"
        artifact_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key,
            tree=artifact_tree,
            project=self.project,
            kind=mlrun.common.schemas.ArtifactCategories.dataset,
            tag=artifact_1_tag,
        )
        artifact_2_body = self._generate_artifact(
            artifact_2_key,
            tree=artifact_tree,
            project=self.project,
            kind=mlrun.common.schemas.ArtifactCategories.dataset.model,
            tag=artifact_2_tag,
        )

        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=self.project,
            tag=artifact_1_tag,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            project=self.project,
            tag=artifact_2_tag,
        )

        artifact_tags = self._db.list_artifact_tags(self._db_session, self.project)
        # latest, v1, v2
        assert len(artifact_tags) == 3
        artifact_tags = self._db.list_artifact_tags(
            self._db_session,
            self.project,
            category=mlrun.common.schemas.ArtifactCategories.dataset,
        )
        assert len(artifact_tags) == 2
        assert artifact_1_tag in artifact_tags
        assert "latest" in artifact_tags
        artifact_tags = self._db.list_artifact_tags(
            self._db_session,
            self.project,
            category=mlrun.common.schemas.ArtifactCategories.model,
        )
        assert len(artifact_tags) == 2
        assert artifact_2_tag in artifact_tags
        assert "latest" in artifact_tags

    def test_store_artifact_restoring_multiple_tags(self):
        artifact_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree_1"
        artifact_2_tree = "artifact_tree_2"
        artifact_1_body = self._generate_artifact(
            artifact_key, tree=artifact_1_tree, project=self.project
        )
        artifact_2_body = self._generate_artifact(
            artifact_key, tree=artifact_2_tree, project=self.project
        )
        artifact_1_tag = "artifact-tag-1"
        artifact_2_tag = "artifact-tag-2"

        # we use deepcopy to avoid changing the original dict
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            copy.deepcopy(artifact_1_body),
            tag=artifact_1_tag,
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            copy.deepcopy(artifact_2_body),
            tag=artifact_2_tag,
            project=self.project,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, tag="*", project=self.project
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
            self._db_session, key=artifact_key, tag=artifact_1_tag, project=self.project
        )
        assert artifact["metadata"]["uid"] == expected_uids[0]
        assert artifact["metadata"]["tag"] == artifact_1_tag
        artifact = self._db.read_artifact(
            self._db_session, key=artifact_key, tag=artifact_2_tag, project=self.project
        )
        assert artifact["metadata"]["uid"] == expected_uids[1]
        assert artifact["metadata"]["tag"] == artifact_2_tag

    def test_store_artifact_with_different_labels(self):
        # create an artifact with a single label
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=self.project
        )
        labels = {"label1": "value1"}
        artifact_1_body["metadata"]["labels"] = {"label1": "value1"}
        artifact_1_body_copy = copy.deepcopy(artifact_1_body)
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            project=self.project,
        )

        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, project=self.project
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
            project=self.project,
        )

        # verify that the artifact has both labels and it didn't create a new artifact
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, project=self.project
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
        artifact_1_key = "artifact_key_1"
        artifact_1_tree = "artifact_tree"
        artifact_1_body = self._generate_artifact(
            artifact_1_key, tree=artifact_1_tree, project=self.project
        )
        artifact_1_tag = "artifact-tag-1"

        artifact_1_uid = self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            tag=artifact_1_tag,
            project=self.project,
        )

        # verify that the artifact has the tag
        artifacts = self._db.list_artifacts(
            self._db_session,
            name=artifact_1_key,
            project=self.project,
            tag=artifact_1_tag,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_1_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_1_tree

        # create a new artifact with the same key and tag, but a different tree
        artifact_2_tree = "artifact_tree_2"
        artifact_2_body = self._generate_artifact(
            artifact_1_key, tree=artifact_2_tree, project=self.project
        )

        artifact_2_uid = self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_2_body,
            tag=artifact_1_tag,
            project=self.project,
        )

        # verify that only the new artifact has the tag
        artifacts = self._db.list_artifacts(
            self._db_session,
            name=artifact_1_key,
            project=self.project,
            tag=artifact_1_tag,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_2_uid
        assert artifacts[0]["metadata"]["tree"] == artifact_2_tree

        # verify that the old artifact is still there, but without the tag
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, project=self.project
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
            project=self.project,
        )
        artifact = self._db.read_artifact(
            self._db_session, key=artifact_different_key, project=self.project
        )
        assert artifact
        assert artifact["metadata"]["key"] == artifact_key

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session, key=artifact_key, project=self.project
            )

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
                project=self.project,
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
                project=self.project,
            )

        # store the artifact with valid db_key which is different than the artifact key
        artifact_body = self._generate_artifact(artifact_valid_key)
        artifact_valid_db_key = "artifact_db_key"
        artifact_body["spec"]["db_key"] = artifact_valid_db_key
        self._db.store_artifact(
            self._db_session,
            artifact_valid_key,
            artifact_body,
            project=self.project,
        )
        artifact = self._db.read_artifact(
            self._db_session, key=artifact_valid_key, project=self.project
        )
        assert artifact
        assert artifact["metadata"]["key"] == artifact_valid_key
        assert artifact["spec"]["db_key"] == artifact_valid_db_key

    def test_store_and_list_artifact_missing_project(self):
        artifact_name = "some-artifact"
        tree = "artifact-tree"
        artifact = self._generate_artifact(artifact_name, tree=tree)

        # store with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            self._db.store_artifact(
                self._db_session,
                key=artifact_name,
                artifact=artifact,
                project=None,
            )

        # store with valid project
        self._db.store_artifact(
            self._db_session,
            key=artifact_name,
            artifact=artifact,
            project=self.project,
        )

        # list with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            self._db.list_artifacts(
                self._db_session,
                project=None,
            )

        # delete with missing project should raise error
        with pytest.raises(mlrun.errors.MLRunMissingProjectError):
            self._db.del_artifacts(self._db_session, project=None)

        self._db.del_artifacts(self._db_session, project=self.project)
        artifacts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
        )
        assert len(artifacts) == 0

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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
            project=self.project,
        )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session,
                key=artifact_1_key,
                tag=artifact_2_tag,
                project=self.project,
            )
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session,
                key=artifact_2_key,
                tag=artifact_1_tag,
                project=self.project,
            )
        # just verifying it's not raising
        self._db.read_artifact(
            self._db_session,
            key=artifact_1_key,
            tag=artifact_1_tag,
            project=self.project,
        )
        self._db.read_artifact(
            self._db_session,
            key=artifact_2_key,
            tag=artifact_2_tag,
            project=self.project,
        )
        # check list
        artifacts = self._db.list_artifacts(
            self._db_session, tag=artifact_1_tag, project=self.project
        )
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(
            self._db_session, tag=artifact_2_tag, project=self.project
        )
        assert len(artifacts) == 1

    def test_overwrite_artifact_with_tag(self):
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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_body_2,
            tag=artifact_2_tag,
            project=self.project,
        )

        identifier_1 = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key,
            uid=artifact_tree,
            iter=0,
        )

        # overwrite the tag for only one of the artifacts
        self._db.overwrite_artifacts_with_tag(
            self._db_session, self.project, "new-tag", [identifier_1]
        )

        # verify that only the first artifact is with the new tag now
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="new-tag"
        )
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag=artifact_1_tag
        )
        assert len(artifacts) == 0

        # verify that the second artifact's tag did not change
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag=artifact_2_tag
        )
        assert len(artifacts) == 1

    def test_modify_artifact_tags(self):
        artifact_key = "artifact-key"
        artifact_body = self._generate_artifact(artifact_key, project=self.project)

        # Step 1: Store artifacts with initial tags "v1" and "v2"
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_body,
            project=self.project,
            tag="v1",
        )
        self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_body,
            project=self.project,
            tag="v2",
        )

        # Verify initial state: 3 artifacts, with the "latest", "v1", and "v2" tags
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == 3
        assert (
            artifacts[0]["metadata"]["tag"]
            == mlrun.common.constants.RESERVED_TAG_NAME_LATEST
        )
        assert artifacts[1]["metadata"]["tag"] == "v2"
        assert artifacts[2]["metadata"]["tag"] == "v1"

        # Step 2: Overwrite artifact with tag "v3"
        identifier = mlrun.common.schemas.ArtifactIdentifier(key=artifact_key)
        self._db.overwrite_artifacts_with_tag(
            self._db_session, self.project, tag="v3", identifiers=[identifier]
        )

        # Verify after overwrite: "latest" remains, all other tags are deleted, and "v3" is added
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == 2
        assert (
            artifacts[0]["metadata"]["tag"]
            == mlrun.common.constants.RESERVED_TAG_NAME_LATEST
        )
        assert artifacts[1]["metadata"]["tag"] == "v3"

        # Step 3: Append tag "v4"
        self._db.append_tag_to_artifacts(
            self._db_session, self.project, tag="v4", identifiers=[identifier]
        )

        # Verify after append: "latest" and "v3" remain, "v4" is added, so we expect 3 artifacts in total
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == 3
        assert (
            artifacts[0]["metadata"]["tag"]
            == mlrun.common.constants.RESERVED_TAG_NAME_LATEST
        )
        assert artifacts[1]["metadata"]["tag"] == "v4"
        assert artifacts[2]["metadata"]["tag"] == "v3"

        # Step 4: Delete tag "v3"
        self._db.delete_tag_from_artifacts(
            self._db_session, self.project, tag="v3", identifiers=[identifier]
        )

        # Verify that "latest" and "v4" tags remain, and "v3" tag is deleted
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == 2
        assert (
            artifacts[0]["metadata"]["tag"]
            == mlrun.common.constants.RESERVED_TAG_NAME_LATEST
        )
        assert artifacts[1]["metadata"]["tag"] == "v4"

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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
            project=self.project,
        )
        self._db.del_artifacts(
            self._db_session, tag=artifact_1_tag, project=self.project
        )
        artifacts = self._db.list_artifacts(
            self._db_session, tag=artifact_1_tag, project=self.project
        )
        assert len(artifacts) == 0
        artifacts = self._db.list_artifacts(
            self._db_session, tag=artifact_2_tag, project=self.project
        )
        assert len(artifacts) == 1
        self._db.del_artifacts(
            self._db_session, tag=artifact_2_tag, project=self.project
        )
        artifacts = self._db.list_artifacts(
            self._db_session, tag=artifact_2_tag, project=self.project
        )
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
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            tag=artifact_2_tag,
            project=self.project,
        )
        with (
            unittest.mock.patch.object(
                self._db,
                "_delete",
                side_effect=mlrun.errors.MLRunInternalServerError("some error"),
            ),
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            self._db.del_artifacts(self._db_session, project=self.project)
        assert "Failed to delete 2 artifacts" in str(exc.value)

        with (
            unittest.mock.patch.object(
                self._db,
                "_delete",
                side_effect=[mlrun.errors.MLRunInternalServerError("some error"), None],
            ),
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            self._db.del_artifacts(self._db_session, project=self.project)
        assert "Failed to delete 1 artifacts" in str(exc.value)

        artifacts = self._db.list_artifacts(
            self._db_session, as_records=True, project=self.project
        )
        assert len(artifacts) == 2
        self._db.del_artifacts(self._db_session, project=self.project)
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 0

    def test_delete_artifacts_exceeds_max_allowed_deletions(self):
        artifact_key = "artifact_key"
        artifact_body = self._generate_artifact(artifact_key)

        # Store two artifacts with the same project and key
        self._db.store_artifact(
            self._db_session,
            key=artifact_key,
            project=self.project,
            iter=0,
            artifact=artifact_body,
        )
        self._db.store_artifact(
            self._db_session,
            key=artifact_key,
            project=self.project,
            iter=1,
            artifact=artifact_body,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == 2

        mlrun.mlconf.artifacts.limits.max_deletions = 1

        with (
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            self._db.del_artifacts(
                self._db_session, project=self.project, name=artifact_key
            )
        assert (
            "Cannot delete 2 artifacts. The maximum allowed artifacts deletions"
            in str(exc.value)
        )

    def test_delete_artifacts_with_specific_iteration(self):
        artifact_key = "artifact_key"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, project=self.project
        )
        num_of_iterations = 5

        # create artifacts with the same key and different iterations
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=self.project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # make sure all artifacts were created
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations

        # delete the artifact with iteration 3
        self._db.del_artifact(
            self._db_session,
            project=self.project,
            key=artifact_key,
            iter=3,
            tag="latest",
        )

        # make sure the artifact with iteration 3 was deleted
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations - 1

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session, key=artifact_key, project=self.project, iter=3
            )

    def test_delete_artifacts_with_specific_uid(self):
        artifact_key = "artifact_key"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, project=self.project
        )
        num_of_iterations = 3

        # create artifacts with the same key and different iterations
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=self.project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # make sure all artifacts were created
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations

        # take the uid of the first artifact
        uid = artifacts[0]["metadata"]["uid"]

        # delete the artifact with the specific uid
        self._db.del_artifact(
            self._db_session, project=self.project, key=artifact_key, uid=uid
        )

        # make sure the artifact with the specific uid was deleted
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_key
        )
        assert len(artifacts) == num_of_iterations - 1

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._db.read_artifact(
                self._db_session, key=artifact_key, project=self.project, uid=uid
            )

    def test_delete_artifact_tag_filter(self):
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
                project=self.project,
            )

        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_1_key
        )
        # Should return 2 tags ('latest' and artifact_1_tag)
        assert len(artifacts) == 2
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag=artifact_2_tag
        )
        assert len(artifacts) == 1
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag=artifact_2_tag_2
        )
        assert len(artifacts) == 1

        self._db.del_artifact(
            self._db_session,
            key=artifact_1_key,
            project=self.project,
            tag=artifact_1_tag,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_1_key, project=self.project
        )
        assert len(artifacts) == 0

        # Negative test - wrong tag, no deletions
        self._db.del_artifact(
            self._db_session,
            key=artifact_2_key,
            project=self.project,
            tag=artifact_1_tag,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_2_key
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

        tags = self._db.list_artifact_tags(self._db_session, self.project)
        assert len(tags) == 3

        # Delete the artifact object (should delete all tags of the same artifact object)
        self._db.del_artifact(
            self._db_session,
            key=artifact_2_key,
            tag=artifact_2_tag_2,
            project=self.project,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_2_key
        )
        assert len(artifacts) == 0

        # Assert all tags were deleted
        tags = self._db.list_artifact_tags(self._db_session, self.project)
        assert len(tags) == 0

    def test_delete_artifact_with_latest_tag_and_iteration_0(
        self,
    ):
        # This test is based on the following scenario:
        # 1. Log an artifact hyperparameters - iteration 1, best_iteration=True
        # 2. Log an artifact hyperparameters - iteration 2, best_iteration=False
        # 3. Log an artifact without hyperparameters - iteration 0, best_iteration=True
        # 4. Delete the artifact with the "latest" tag - the last artifact that was logged (iteration 0)
        # 5. The "latest" tag should move to both iteration artifacts of the hyperparameter run

        artifact_key = "artifact-key"
        artifact_1_tree = "artifact_tree_1"
        artifact_1_body = self._generate_artifact(artifact_key, tree=artifact_1_tree)
        artifact_2_tree = "artifact_tree_2"
        artifact_2_body = self._generate_artifact(artifact_key, tree=artifact_2_tree)

        # Log the first artifact as part of a function run with hyperparameters (iteration 1, best_iteration=True)
        uid1 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_1_body,
            project=self.project,
            tag="v1",
            iter=1,
            best_iteration=True,
        )

        # Log the second artifact as part of a function run with hyperparameters (iteration 2, without best_iteration)
        uid2 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_1_body,
            project=self.project,
            tag="v2",
            iter=2,
        )

        # Logging artifact in a regular way, with iteration = 0 best_iteration=True.
        # The "latest" tag should now be assigned to this artifact.
        uid3 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_2_body,
            project=self.project,
            tag="v3",
            best_iteration=True,
        )

        assert uid1 != uid2 != uid3
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project
        )
        # When logging an artifact with hyperparameters, each artifact also receives the 'latest' tag
        # (v1, latest, v2, latest). However, when logging the artifact without hyperparameters, the 'latest' tag
        # moves to this artifact and is removed from the others (v1, v2, v3, latest).
        assert len(artifacts) == 4

        # Verify that the "latest" tag is correctly attached to the artifact with uid3
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == uid3
        assert artifacts[0]["metadata"]["tag"] == "latest"

        # Delete the artifact that currently holds the "latest" tag (uid3)
        self._db.del_artifact(
            self._db_session, key=artifact_key, project=self.project, tag="latest"
        )

        # The "latest" tag should move to the most recent artifacts
        # This should be both iterations of the hyperparameter run (uid1 and uid2)
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        assert len(artifacts) == 2
        assert sorted(
            [artifact["metadata"]["uid"] for artifact in artifacts]
        ) == sorted([uid1, uid2])
        assert all(artifact["metadata"]["tag"] == "latest" for artifact in artifacts)

    def test_delete_artifact_with_latest_tag_and_iteration_not_0(self):
        # This test is based on the following scenario:
        # 1. Log 3 artifacts with hyperparameters - iteration 1 (best_iteration), iteration 2, and iteration 3.
        # 2. Log 2 artifacts with hyperparameters (same artifacts, but fewer iterations) - iteration 1 and iteration 2.
        # 3. Delete an artifact from the second run (iteration 2).
        # 4. The "latest" tag should not move because there is still an artifact with the latest tag in other
        #    iterations with the same producer id.
        # 5. Delete an artifact from the first run (iteration 3).
        # 6. The "latest" tag should not move, because artifact is not holding the latest tag (artifact is untagged).
        # 7. Delete the last artifact from the second run (iteration 1).
        # 8. The "latest" tag should move because there is no other latest tag in the same producer id
        #    for other iterations.
        #    move the latest tag to all remaining iterations of the previous latest run.

        artifact_key = "artifact-key"
        artifact_1_tree = "artifact_tree_1"
        artifact_1_body = self._generate_artifact(artifact_key, tree=artifact_1_tree)
        artifact_2_tree = "artifact_tree_2"
        artifact_2_body = self._generate_artifact(artifact_key, tree=artifact_2_tree)

        # Log the first artifact as part of a function run with hyperparameters (iteration 1, best_iteration=True)
        uid1 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_1_body,
            project=self.project,
            tag="v1",
            iter=1,
            best_iteration=True,
        )

        # Log the second artifact as part of a function run with hyperparameters (iteration 2, without best_iteration)
        uid2 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_1_body,
            project=self.project,
            tag="v1",
            iter=2,
        )

        # Log the third artifact as part of a function run with hyperparameters (iteration 3, without best_iteration)
        uid3 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_1_body,
            project=self.project,
            tag="v1",
            iter=3,
        )

        assert uid1 != uid2 != uid3

        # Should have both "v1" and "latest" tags for each of the artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project
        )
        assert len(artifacts) == 6

        # Log the same function again with hyperparameters, but now only with 2 iterations (iteration 1 and iteration 2)
        uid4 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_2_body,
            project=self.project,
            tag="v1",
            iter=1,
        )

        uid5 = self._db.store_artifact(
            self._db_session,
            artifact_key,
            artifact_2_body,
            project=self.project,
            tag="v1",
            iter=2,
            best_iteration=True,
        )
        assert uid1 != uid2 != uid3 != uid4 != uid5
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        # All the artifacts from previous runs are now untagged.
        assert len(artifacts) == 2
        assert artifacts[0]["metadata"]["uid"] == uid5
        assert artifacts[1]["metadata"]["uid"] == uid4

        # Delete artifact uid5
        self._db.del_artifact(
            self._db_session, key=artifact_key, project=self.project, uid=uid5
        )

        # The "latest" tag should not be moved, as there is still an artifact in other iterations with
        # the "latest" tag and the same producer ID.
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == uid4

        # Delete artifact uid3 (which does not have the "latest" tag) - The "latest" tag should not be moved.
        self._db.del_artifact(
            self._db_session, key=artifact_key, project=self.project, uid=uid3
        )
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == uid4

        # Delete artifact uid4
        self._db.del_artifact(
            self._db_session, key=artifact_key, project=self.project, uid=uid4
        )

        # The "latest" tag should be moved because there is no other "latest" tag for the same producer ID in
        # other iterations. Moved to all remaining iterations of the previous latest run.
        artifacts = self._db.list_artifacts(
            self._db_session, name=artifact_key, project=self.project, tag="latest"
        )
        assert len(artifacts) == 2
        assert sorted(
            [artifact["metadata"]["uid"] for artifact in artifacts]
        ) == sorted([uid1, uid2])

    def test_delete_artifacts_in_batches(self):
        artifact_key_prefix = "artifact_key"
        artifact_body = self._generate_artifact(artifact_key_prefix)

        # Store artifacts
        for i in range(15):
            self._db.store_artifact(
                self._db_session,
                key=f"{artifact_key_prefix}_{i}",
                project=self.project,
                iter=0,
                artifact=artifact_body,
            )

        # Verify artifacts were stored
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 15

        # Set small batch size to force batch deletion
        mlrun.mlconf.httpdb.projects.resource_deletion_batch_size = 5

        where_clause = ArtifactV2.project == self.project

        with unittest.mock.patch.object(
            self._db_session, "execute", wraps=self._db_session.execute
        ) as mock_execute:
            deleted_count = SQLDB._delete_table_in_batches(
                self._db_session,
                ArtifactV2,
                where_clause,
            )
            delete_calls = [
                call
                for call in mock_execute.call_args_list
                if str(call[0][0]).startswith("DELETE")
            ]
            assert len(delete_calls) == 3, (
                f"Expected 3 batch deletions, got {len(delete_calls)}"
            )

        # Validate that all artifacts were deleted
        assert deleted_count == 15

        artifacts_after_deletion = self._db.list_artifacts(
            self._db_session, project=self.project
        )
        assert len(artifacts_after_deletion) == 0

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
            project=self.project,
        )
        artifact_1_body["iter"] = 42
        self._db.store_artifact(
            self._db_session,
            artifact_1_key,
            artifact_1_body,
            iter=42,
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            project=self.project,
        )
        artifact_2_body["iter"] = 42
        self._db.store_artifact(
            self._db_session,
            artifact_2_key,
            artifact_2_body,
            iter=42,
            project=self.project,
        )

        def _list_and_assert_count(key, count, iter=None):
            results = self._db.list_artifacts(
                self._db_session, name=key, iter=iter, project=self.project
            )
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
        tag = "mytag1"

        self._generate_artifact_with_iterations(
            artifact_key_1,
            artifact_tree_1,
            num_iters,
            best_iter,
            ArtifactCategories.model,
            project=self.project,
        )

        self._generate_artifact_with_iterations(
            artifact_key_2,
            artifact_tree_2,
            num_iters,
            best_iter,
            ArtifactCategories.model,
            project=self.project,
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
            self._db_session, self.project, tag, [identifier_1, identifier_2]
        )
        results = self._db.list_artifacts(
            self._db_session, project=self.project, tag=tag, best_iteration=True
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
            project=self.project,
        )
        self._generate_artifact_with_iterations(
            artifact_2_key,
            artifact_2_tree,
            num_iters,
            best_iter_2,
            ArtifactCategories.dataset,
            project=self.project,
        )

        # Add non-hyper-param artifact. Single object with iter 0, not pointing at anything
        artifact_body = self._generate_artifact(
            artifact_no_link_key, artifact_no_link_tree
        )
        artifact_body["spec"]["iter"] = 0
        self._db.store_artifact(
            self._db_session,
            artifact_no_link_key,
            artifact_body,
            iter=0,
            project=self.project,
        )

        results = self._db.list_artifacts(
            self._db_session, name="~artifact", project=self.project
        )
        # we don't store link artifacts in the DB, so we expect 2 * num_iters - 1, plus a regular artifact
        assert len(results) == (num_iters - 1) * 2 + 1

        results = self._db.list_artifacts(
            self._db_session,
            name=artifact_1_key,
            best_iteration=True,
            project=self.project,
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        expected_iters = {
            artifact_1_key: best_iter_1,
            artifact_2_key: best_iter_2,
            artifact_no_link_key: 0,
        }
        results = self._db.list_artifacts(
            self._db_session,
            name="~artifact",
            best_iteration=True,
            project=self.project,
        )
        assert len(results) == 3
        for artifact in results:
            artifact_name = artifact["metadata"]["key"]
            assert (
                artifact_name in expected_iters
                and expected_iters[artifact_name] == artifact["spec"]["iter"]
            )

        results = self._db.list_artifacts(
            self._db_session,
            best_iteration=True,
            category=ArtifactCategories.model,
            project=self.project,
        )
        assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

        # Should get only object-2 (which is of dataset type) without the link artifact
        results = self._db.list_artifacts(
            self._db_session, category=ArtifactCategories.dataset, project=self.project
        )
        assert len(results) == num_iters - 1
        for artifact in results:
            assert artifact["metadata"]["key"] == artifact_2_key

        # Negative test - asking for both best_iter and iter
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            results = self._db.list_artifacts(
                self._db_session,
                name="~artifact",
                best_iteration=True,
                iter=0,
                project=self.project,
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
            project=self.project,
        )
        self._generate_artifact_with_iterations(
            artifact_key,
            artifact_2_tree,
            num_iters,
            best_iter_2,
            ArtifactCategories.model,
            project=self.project,
        )
        self._generate_artifact_with_iterations(
            artifact_key,
            artifact_3_tree,
            num_iters,
            best_iter_3,
            ArtifactCategories.model,
            project=self.project,
        )

        for category in [ArtifactCategories.model, None]:
            results = self._db.list_artifacts(
                self._db_session,
                tag="*",
                best_iteration=True,
                category=category,
                project=self.project,
            )
            assert len(results) == 3
            for result in results:
                if result["metadata"]["tree"] == artifact_3_tree:
                    assert result["metadata"].get("tag") == "latest"
                else:
                    assert not result["metadata"].get("tag")

    def test_list_artifact_for_tagging_fallback(self):
        # create an artifact
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
            project=self.project,
        )

        # append artifact tag, but put the `tree` in the `uid` field of the identifier, like older clients do
        identifier = mlrun.common.schemas.ArtifactIdentifier(
            kind=ArtifactCategories.model,
            key=artifact_key,
            uid=artifact_tree,
        )
        artifact_tag_2 = "artifact-tag-2"
        self._db.append_tag_to_artifacts(
            self._db_session, self.project, artifact_tag_2, [identifier]
        )

        # verify that the artifact has both tags
        artifacts = self._db.list_artifacts(
            self._db_session,
            name=artifact_key,
            project=self.project,
            tag=artifact_tag_1,
        )
        assert len(artifacts) == 1

        artifacts = self._db.list_artifacts(
            self._db_session,
            name=artifact_key,
            project=self.project,
            tag=artifact_tag_2,
        )
        assert len(artifacts) == 1

    @pytest.mark.parametrize("limit", [None, 6])
    def test_list_artifacts_returns_elements_by_order_updated_field(self, limit):
        artifact_kinds = ArtifactCategories.all()

        # Create artifacts
        number_of_artifacts = 10
        for counter in range(number_of_artifacts):
            next_cyclic_item = artifact_kinds[counter % len(artifact_kinds)]
            artifact_key = f"artifact-{counter}"
            artifact_body = self._generate_artifact(
                artifact_key, project=self.project, kind=next_cyclic_item
            )
            self._db.store_artifact(
                self._db_session, artifact_key, artifact_body, project=self.project
            )

        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, limit=limit
        )

        expected_count = limit or number_of_artifacts
        assert len(artifacts) == expected_count, (
            f"Expected {expected_count} results, got {len(artifacts)}"
        )

        start_index = number_of_artifacts - 1
        expected_names = [
            f"artifact-{i}"
            for i in range(start_index, start_index - expected_count, -1)
        ]

        for artifact, expected_name in zip(artifacts, expected_names):
            artifact_name = artifact["metadata"]["key"]
            assert artifact_name == expected_name, (
                f"Expected {expected_name}, got {artifact_name}"
            )

    @pytest.mark.parametrize("limit", [None, 6])
    def test_list_artifacts_orders_by_id_when_updated_is_identical(self, limit):
        # this test is verified that when updated date is identical, artifacts should be ordered by artifact id

        t1 = datetime.datetime.now()

        # Create artifacts
        number_of_artifacts = 10
        for counter in range(number_of_artifacts):
            artifact_key = f"artifact-{counter}"
            artifact_body = self._generate_artifact(
                artifact_key,
                project=self.project,
                labels={"key1": "val1", "key2": "val2"},
            )
            self._db.store_artifact(
                self._db_session, artifact_key, artifact_body, project=self.project
            )

            # Set the same `updated` timestamp for all artifacts
            self._db.update_db_object(
                self._db_session,
                framework.db.sqldb.models.ArtifactV2,
                filters={"key": artifact_key},
                updated=t1,
            )

        # We are also listing with labels to verify that ordering works correctly with labels and limit.
        artifacts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
            limit=limit,
            labels="key1=val1",
        )

        expected_count = limit or number_of_artifacts
        assert len(artifacts) == expected_count, (
            f"Expected {expected_count} results, got {len(artifacts)}"
        )

        start_index = number_of_artifacts - 1
        expected_names = [
            f"artifact-{i}"
            for i in range(start_index, start_index - expected_count, -1)
        ]

        for artifact, expected_name in zip(artifacts, expected_names):
            artifact_name = artifact["metadata"]["key"]
            assert artifact_name == expected_name, (
                f"Expected {expected_name}, got {artifact_name}"
            )

    @pytest.mark.parametrize("limit", [None, 3])
    @pytest.mark.parametrize("tag", [None, "*"])
    def test_list_artifacts_orders_by_tag_id(self, limit, tag):
        # This test verifies that when an artifact has multiple tags, the returned list is ordered with 'latest'
        # first and the rest by tag ID descending.

        artifact_key = "dummy-artifact"

        artifact_body = self._generate_artifact(
            key=artifact_key,
            project=self.project,
        )

        number_of_tags = 5
        for counter in range(number_of_tags):
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=self.project,
                tag=f"v{counter}",
            )

        artifacts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
            limit=limit,
            tag=tag,
        )

        expected_count = limit or (number_of_tags + 1)  # one more for latest tag

        # Build expected tag order with "latest" first
        expected_tags = [mlrun.common.constants.RESERVED_TAG_NAME_LATEST] + [
            f"v{i}" for i in reversed(range(number_of_tags))
        ]
        expected_tags = expected_tags[:expected_count]

        actual_tags = [artifact["metadata"]["tag"] for artifact in artifacts]
        assert actual_tags == expected_tags, (
            f"Expected tags {expected_tags}, got {actual_tags}"
        )

        # Verify the case of listing artifacts by a specific tag, which should result in an inner join and
        # return only the matching tagged artifact
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, limit=limit, tag="v3"
        )
        assert len(artifacts) == 1

        # List artifacts partitioned by 'project' and 'name' to verify that the query is working as expected
        # when using both 'partition_by' and 'order_by'.
        # The test verifies the query behavior both with and without the 'limit' parameter.
        artifacts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
            limit=limit,
            partition_by=mlrun.common.schemas.ArtifactPartitionByField.project_and_name,
        )
        assert len(artifacts) == 1
        assert (
            artifacts[0]["metadata"]["tag"]
            == mlrun.common.constants.RESERVED_TAG_NAME_LATEST
        )

    def test_list_artifacts_producer_uri(self):
        artifact_key = "dummy-artifact"

        def store_artifact_with_producer(artifact_key, project, producer_uri, tag):
            producer = {"uri": producer_uri}
            artifact_body = self._generate_artifact(
                artifact_key, project=project, producer=producer, tag=tag
            )
            self._db.store_artifact(
                self._db_session, artifact_key, artifact_body, project=project
            )

        producer_uri_without_iteration = f"{self.project}/dummy-run-id"
        first_producer_uri = f"{producer_uri_without_iteration}-0"
        store_artifact_with_producer(
            artifact_key, self.project, first_producer_uri, tag="v1"
        )

        second_producer_uri = f"{producer_uri_without_iteration}-1"
        store_artifact_with_producer(
            artifact_key, self.project, second_producer_uri, tag="v2"
        )

        artifacts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
            producer_uri=producer_uri_without_iteration,
        )

        assert len(artifacts) == 2, f"Expected 2 artifacts, but found {len(artifacts)}"
        assert artifacts[0]["spec"]["producer"]["uri"] == second_producer_uri, (
            f"Expected producer URI {second_producer_uri}, but got {artifacts[0]['spec']['producer']['uri']}"
        )

        assert artifacts[1]["spec"]["producer"]["uri"] == first_producer_uri, (
            f"Expected producer URI {first_producer_uri}, but got {artifacts[1]['spec']['producer']['uri']}"
        )

    def test_iterations_with_latest_tag(self):
        artifact_key = "artifact_key"
        artifact_tree = "artifact_tree"
        artifact_body = self._generate_artifact(
            artifact_key, tree=artifact_tree, project=self.project
        )
        num_of_iterations = 5

        # create artifacts with the same key and different iterations
        for iteration in range(1, num_of_iterations + 1):
            artifact_body["metadata"]["iter"] = iteration
            self._db.store_artifact(
                self._db_session,
                artifact_key,
                artifact_body,
                project=self.project,
                iter=iteration,
                producer_id=artifact_tree,
            )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations

        # mark iteration 3 as the best iteration
        best_iteration = 3
        self._mark_best_iteration_artifact(
            self.project, artifact_key, artifact_tree, best_iteration
        )

        # list artifact with "latest" tag - should return all artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations

        # list artifact with "latest" tag and best_iteration=True - should return only the artifact with iteration 3
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest", best_iteration=True
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
                project=self.project,
                iter=iteration,
                producer_id=artifact_tree_2,
            )

        # list artifact with "latest" tag - should return only the new artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert len(artifacts) == num_of_iterations
        producer_ids = set([artifact["metadata"]["tree"] for artifact in artifacts])
        assert len(producer_ids) == 1
        assert producer_ids.pop() == artifact_tree_2

        # mark iteration 2 as the best iteration
        best_iteration = 2
        self._mark_best_iteration_artifact(
            self.project, artifact_key, artifact_tree_2, best_iteration
        )

        # list artifact with "latest" tag and best iteration - should return only the new artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest", best_iteration=True
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["iter"] == best_iteration
        assert artifacts[0]["metadata"]["tree"] == artifact_tree_2

    @pytest.mark.asyncio
    async def test_project_file_counter(self):
        # create artifact with 5 distinct keys, and 3 tags for each key
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
                    project=self.project,
                    producer_id=artifact_tree,
                )

        # list artifact with "latest" tag - should return 5 artifacts
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert len(artifacts) == 5

        # query all artifacts tags, should return 4 tags = 3 tags + latest
        tags = self._db.list_artifact_tags(self._db_session, project=self.project)
        assert len(tags) == 4

        # files counters should return the most recent artifacts, for each key -> 5 artifacts
        project_to_files_count = self._db._calculate_artifact_counters_by_category(
            self._db_session
        )[mlrun.common.schemas.ArtifactCategories.other]
        assert project_to_files_count[self.project] == 5

    def test_update_model_spec(self):
        artifact_key = "model1"

        # create a model
        model_body = self._generate_artifact(artifact_key, kind="model")
        self._db.store_artifact(
            self._db_session, artifact_key, model_body, project=self.project
        )
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update the model with spec that should be ignored in UID calc
        model_body["spec"]["parameters"] = {"p1": 5}
        model_body["spec"]["outputs"] = {"o1": 6}
        model_body["spec"]["metrics"] = {"l1": "a"}
        self._db.store_artifact(
            self._db_session, artifact_key, model_body, project=self.project
        )
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # update spec that should not be ignored
        model_body["spec"]["model_file"] = "some/path"
        self._db.store_artifact(
            self._db_session, artifact_key, model_body, project=self.project
        )
        artifacts = self._db.list_artifacts(self._db_session, project=self.project)
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
        self._db.store_artifact(
            self._db_session,
            k1,
            art1,
            producer_id=t1,
            iter=1,
            project=self.project,
            tag="tag1",
        )
        self._db.store_artifact(
            self._db_session,
            k1,
            art2,
            producer_id=t2,
            iter=2,
            project=self.project,
            tag="tag2",
        )

        result = self._db.read_artifact(
            self._db_session, key=k1, tag="tag1", iter=1, project=self.project
        )
        assert result["metadata"]["tag"] == "tag1"
        result = self._db.read_artifact(
            self._db_session, key=k1, tag="tag2", iter=2, project=self.project
        )
        assert result["metadata"]["tag"] == "tag2"
        result = self._db.read_artifact(
            self._db_session, key=k1, iter=1, project=self.project
        )
        # When doing get without a tag, the returned object must not contain a tag.
        assert "tag" not in result["metadata"]

        result = self._db.list_artifacts(
            self._db_session, name=k1, project=self.project, tag="*"
        )
        assert len(result) == 3
        for artifact in result:
            assert (
                (artifact["a"] == 1 and artifact["metadata"]["tag"] == "tag1")
                or (artifact["a"] == 2 and artifact["metadata"]["tag"] == "tag2")
                or (artifact["a"] in (1, 2) and artifact["metadata"]["tag"] == "latest")
            )

        # To be used later, after adding tags
        full_results = result

        result = self._db.list_artifacts(
            self._db_session, name=k1, tag="tag1", project=self.project
        )
        assert (
            len(result) == 1
            and result[0]["metadata"]["tag"] == "tag1"
            and result[0]["a"] == 1
        )
        result = self._db.list_artifacts(
            self._db_session, name=k1, tag="tag2", project=self.project
        )
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
            self._db_session,
            artifacts,
            self.project,
            name=new_tag,
            obj_name_attribute="key",
        )
        result = self._db.list_artifacts(
            self._db_session, name=k1, project=self.project, tag="*"
        )
        assert deepdiff.DeepDiff(result, expected_results, ignore_order=True) == {}

        # Add another tag to the art1
        self._db.store_artifact(
            self._db_session,
            k1,
            art1,
            producer_id=t1,
            iter=1,
            project=self.project,
            tag="tag3",
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
            self._db_session, key=k1, tag="tag3", iter=1, project=self.project
        )
        assert result["metadata"]["tag"] == "tag3"
        expected_results.append(copy.deepcopy(result))

        # add it again but with the "latest" tag
        result["metadata"]["tag"] = "latest"
        expected_results.append(result)

        result = self._db.list_artifacts(
            self._db_session, name=k1, project=self.project, tag="*"
        )
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
        self._db.store_artifact(
            self._db_session, k1, art1, producer_id=t1, project=self.project
        )

        arts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert art1["a"] == arts[0]["a"], "bad artifact"

        t2, art2 = "t2", {"a": 17}
        self._db.store_artifact(
            self._db_session, k1, art2, producer_id=t2, project=self.project
        )
        arts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
        assert 1 == len(arts), "count"
        assert art2["a"] == arts[0]["a"], "bad artifact"

        k2, t3, art3 = "k2", "t3", {"a": 99}
        self._db.store_artifact(
            self._db_session, k2, art3, producer_id=t3, project=self.project
        )
        arts = self._db.list_artifacts(
            self._db_session, project=self.project, tag="latest"
        )
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

        # create artifacts in the db directly to avoid the store_artifact function which sets the updated field
        artifacts_to_create = []
        for key, updated, producer_id in [
            ("k1", t1, "p1"),
            ("k2", t2, "p2"),
            ("k3", t3, "p3"),
        ]:
            artifact_struct = mlrun.artifacts.Artifact(
                metadata=mlrun.artifacts.ArtifactMetadata(
                    key=key, project=self.project, tree=producer_id
                ),
                spec=mlrun.artifacts.ArtifactSpec(),
            )
            db_artifact = ArtifactV2(
                project=self.project, key=key, updated=updated, producer_id=producer_id
            )
            db_artifact.full_object = artifact_struct.to_dict()
            artifacts_to_create.append(db_artifact)

        self._db._upsert(self._db_session, artifacts_to_create)

        arts = self._db.list_artifacts(
            self._db_session, project=self.project, since=t3, tag="*"
        )
        assert 3 == len(arts), "since t3"

        arts = self._db.list_artifacts(
            self._db_session, project=self.project, since=t2, tag="*"
        )
        assert 2 == len(arts), "since t2"

        arts = self._db.list_artifacts(
            self._db_session,
            project=self.project,
            since=t1 + datetime.timedelta(days=1),
            tag="*",
        )
        assert not arts, "since t1+"

        arts = self._db.list_artifacts(
            self._db_session, project=self.project, until=t2, tag="*"
        )
        assert 2 == len(arts), "until t2"

        arts = self._db.list_artifacts(
            self._db_session, project=self.project, since=t2, until=t2, tag="*"
        )
        assert 1 == len(arts), "since/until t2"

    def test_list_artifacts_for_producer_id(self):
        artifact_name = "artifact_name"
        tree1 = "artifact_tree_1"
        tree2 = "artifact_tree_2"

        # Generate and store two artifacts with the same name and different producer id
        artifact_1 = self._generate_artifact(artifact_name, tree=tree1)
        artifact_2 = self._generate_artifact(
            artifact_name, project=self.project, tree=tree2
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name,
            artifact_1,
            project=self.project,
        )
        self._db.store_artifact(
            self._db_session,
            artifact_name,
            artifact_2,
            project=self.project,
        )
        artifacts = self._db.list_artifacts(
            self._db_session, project=self.project, name=artifact_name
        )
        assert len(artifacts) == 2

        # Retrieve the first artifact without tag (None) and with specific UID
        artifact_identifiers = [(artifact_name, None, 0, artifact_1["metadata"]["uid"])]
        artifacts = self._db.list_artifacts_for_producer_id(
            self._db_session,
            project=self.project,
            producer_id=tree1,
            artifact_identifiers=artifact_identifiers,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_1["metadata"]["uid"]
        assert artifacts[0]["metadata"]["tag"] is None

        # Retrieve same artifact without tag (None) and without specific UID
        artifact_identifiers = [(artifact_name, None, 0, None)]
        artifacts = self._db.list_artifacts_for_producer_id(
            self._db_session,
            project=self.project,
            producer_id=tree1,
            artifact_identifiers=artifact_identifiers,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_1["metadata"]["uid"]
        assert artifacts[0]["metadata"]["tag"] is None

        # Retrieve the second artifact with tag "latest" and no UID
        artifact_identifiers = [(artifact_name, None, 0, None)]
        artifacts = self._db.list_artifacts_for_producer_id(
            self._db_session,
            project=self.project,
            producer_id=tree2,
            artifact_identifiers=artifact_identifiers,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["uid"] == artifact_2["metadata"]["uid"]
        assert artifacts[0]["metadata"]["tag"] == "latest"

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            pytest.param(
                {
                    "limit": 1001,
                    "best_iteration": True,
                    "tag": "latest",
                },
                True,
                id="default_query",
            ),
            pytest.param(
                {
                    "best_iteration": True,
                    "tag": "latest",
                },
                False,
                id="no_pagination",
            ),
            pytest.param(
                {
                    "limit": 1001,
                    "best_iteration": False,
                    "tag": "latest",
                },
                False,
                id="best_iteration_false",
            ),
            pytest.param(
                {
                    "limit": 1001,
                    "best_iteration": True,
                    "tag": "any_tag",
                },
                False,
                id="tag_not_latest",
            ),
            pytest.param(
                {
                    "limit": 1001,
                    "best_iteration": True,
                    "tag": "latest",
                    "name": "any_name",
                },
                False,
                id="additional_params",
            ),
        ],
    )
    def test_is_default_list_artifacts_query(self, kwargs: dict, expected: bool):
        ignored_params = {
            "project": self.project,
            "category": "any_category",
            "offset": 5,  # any offset
        }
        kwargs.update(ignored_params)
        assert self._db._is_default_list_artifacts_query(**kwargs) == expected

    def test_parent_uri_without_tag(self):
        # Create referenced artifact
        parent_artifact_name = "parent-artifact"
        child_artifact_name = "child-artifact"
        project = "test-project"
        parent_artifact = self._generate_artifact(parent_artifact_name)
        parent_artifact_2 = self._generate_artifact(
            parent_artifact_name, tree="parent_artifact_2"
        )

        uid = self._db.store_artifact(
            self._db_session,
            parent_artifact_name,
            parent_artifact,
            project,
        )
        self._db.store_artifact(
            self._db_session,
            parent_artifact_name,
            parent_artifact_2,
            project,
        )

        parent_artifact_db = Artifact.from_dict(
            self._db.read_artifact(
                self._db_session,
                key=parent_artifact_name,
                project=project,
                uid=uid,
            )
        )

        assert parent_artifact_db.metadata.tag is None  # Simulate no tag

        # Create artifact that references the above (manually inject the reference UID)
        child_artifact = self._generate_artifact(child_artifact_name)
        child_artifact["spec"]["parent_uri"] = parent_artifact_db.uri

        self._db.store_artifact(
            self._db_session,
            child_artifact_name,
            child_artifact,
            project,
        )

        child_artifact_db = Artifact.from_dict(
            self._db.read_artifact(
                self._db_session,
                key=child_artifact_name,
                project=project,
            )
        )

        assert ":" not in child_artifact_db.spec.parent_uri.split("://", maxsplit=1)[1]

    def test_list_artifact_parent_filter(self):
        # Create referenced artifact
        parent_artifact_name = "parent-artifact"
        child_artifact_name = "child-artifact"
        project = "test-project"
        parent_artifact = self._generate_artifact(parent_artifact_name)
        self._db.store_artifact(
            self._db_session,
            parent_artifact_name,
            parent_artifact,
            project,
            tag="ref-tag",
        )
        parent_artifact_db = Artifact.from_dict(
            self._db.read_artifact(
                self._db_session,
                key=parent_artifact_name,
                tag="ref-tag",
                project=project,
            )
        )

        # Create artifact that references the above (manually inject the reference UID)
        child_artifact = self._generate_artifact(child_artifact_name)
        child_artifact["spec"]["parent_uri"] = parent_artifact_db.uri

        self._db.store_artifact(
            self._db_session,
            child_artifact_name,
            child_artifact,
            project,
        )

        # Filter using parent_key
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri=parent_artifact_name, project=project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name

        # Filter using partial parent_key
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri="parent-ar", project=project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name

        # Filter using parent_tag
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri=":ref-tag", project=project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name

        # Filter using parent_tag
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri=":lat", project=project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name
        assert "latest" in artifacts[0]["spec"]["parent_uri"]

        # Filter using partial parent_tag
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri=":ref", project=project
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name
        assert "ref-tag" in artifacts[0]["spec"]["parent_uri"]

        # Filter using both name and tag
        artifacts = self._db.list_artifacts(
            self._db_session,
            parent_uri=f"{parent_artifact_name}:ref-tag",
            project=project,
        )
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == child_artifact_name

        # Negative case
        artifacts = self._db.list_artifacts(
            self._db_session, parent_uri="nonexistent", project=project
        )
        assert len(artifacts) == 0

        artifact = self._db.read_artifact(
            self._db_session, key=parent_artifact_name, project=project
        )

        assert artifact["spec"]["has_children"]

        c_artifact = self._db.read_artifact(
            self._db_session, key=child_artifact_name, project=project
        )

        assert c_artifact["spec"]["parent_uri"] == parent_artifact_db.get_store_url()

    def test_delete_parent_artifacts(self):
        # Create referenced artifact
        parent_artifact_name = "parent-artifact"
        child_artifact_name = "child-artifact"
        project = "test-project"
        parent_artifact = self._generate_artifact(parent_artifact_name)
        self._db.store_artifact(
            self._db_session,
            parent_artifact_name,
            parent_artifact,
            project,
            tag="ref-tag",
        )
        parent_artifact_db = Artifact.from_dict(
            self._db.read_artifact(
                self._db_session,
                key=parent_artifact_name,
                tag="ref-tag",
                project=project,
            )
        )

        # Create artifact that references the above (manually inject the reference UID)
        child_artifact = self._generate_artifact(child_artifact_name)
        child_artifact["spec"]["parent_uri"] = parent_artifact_db.uri

        self._db.store_artifact(
            self._db_session,
            child_artifact_name,
            child_artifact,
            project,
        )
        with pytest.raises(mlrun.errors.MLRunConflictError):
            # delete the parent artifacts
            self._db.del_artifacts(
                self._db_session,
                name=parent_artifact_name,
                project=project,
            )
        self._db.del_artifact(
            self._db_session,
            key=child_artifact_name,
            project=project,
        )
        self._db.del_artifacts(
            self._db_session,
            name=parent_artifact_name,
            project=project,
        )

        artifacts = self._db.list_artifacts(session=self._db_session, project=project)
        assert artifacts == []

    @pytest.mark.parametrize(
        "case",
        [
            {"with_entities": None, "attach_tags": False, "expected": True},
            {"with_entities": None, "attach_tags": True, "expected": True},
            {
                "ids": ["non-empty"],
                "with_entities": None,
                "attach_tags": False,
                "expected": False,
            },
            {"ids": [], "with_entities": None, "attach_tags": False, "expected": True},
            {"ids": [], "with_entities": [], "attach_tags": False, "expected": True},
        ],
        ids=[
            "default-no-ids",
            "default-with-attach-tags",
            "non-default-with-ids",
            "default-empty-ids-none-entities",
            "default-empty-ids-empty-entities",
        ],
    )
    def test_is_default_list_artifacts_query_defaults(self, case):
        """
        Verify the predicate returns True only for the exact UI default list-artifacts query.
        Pass caller-side UI defaults + only the fields under test.
        """
        kwargs = {}
        for key in ("ids", "with_entities", "attach_tags"):
            if key in case:
                kwargs[key] = case[key]

        result = self._db._is_default_list_artifacts_query(
            project=self.project,
            **self._ui_defaults(),
            **kwargs,
        )

        assert result == case["expected"], f"Unexpected result for case {case}"

    @pytest.mark.parametrize(
        "scenario, ui_overrides, expect_hint",
        [
            ("ui-default", {}, True),
            (
                "partition_by-name",
                {"partition_by": mlrun.common.schemas.ArtifactPartitionByField.name},
                False,
            ),
            (
                "sort-order-asc",
                {"partition_order": mlrun.common.schemas.OrderType.asc},
                False,
            ),
            (
                "sort-by-created",
                {"partition_sort_by": mlrun.common.schemas.SortField.created},
                False,
            ),
            ("limit-50", {"limit": 50}, False),
            ("non-latest-tag", {"tag": "v1"}, False),
            ("best_iteration-false", {"best_iteration": False}, False),
            ("ids-non-empty", {"ids": ["force-non-default"]}, False),
            ("ids-empty-list", {"ids": []}, True),
            ("with_entities-minimal", {"with_entities": []}, True),
            ("attach_tags-true", {"attach_tags": True}, True),
        ],
    )
    def test_mysql_use_index_hint_scoping(
        self, monkeypatch, scenario, ui_overrides, expect_hint
    ):
        """
        USE INDEX should be applied ONLY for the exact UI default shape.
        Any deviation should NOT get the hint.
        """
        key = "artifact-for-default-query"
        self._db.store_artifact(
            self._db_session,
            key=key,
            artifact=self._generate_artifact(key, project=self.project),
            project=self.project,
        )

        hint_called = {"value": False}
        real_with_hint = Query.with_hint

        def with_hint_spy(q, selectable, text, dialect_name=None):
            if "USE INDEX" in str(text):
                hint_called["value"] = True
            return real_with_hint(q, selectable, text, dialect_name=dialect_name)

        monkeypatch.setattr(Query, "with_hint", with_hint_spy, raising=True)

        kwargs = {"project": self.project, **self._ui_defaults(), **ui_overrides}
        _ = self._db._find_artifacts(self._db_session, **kwargs)

        if expect_hint:
            assert hint_called["value"], f"{scenario}: expected USE INDEX hint"
        else:
            assert not hint_called["value"], (
                f"{scenario}: did NOT expect USE INDEX hint"
            )

    @pytest.mark.parametrize(
        "scenario, attach_tags, ids_value, expect_hint",
        [
            ("default-query-attach-tags-false", False, None, True),
            ("default-query-attach-tags-true", True, None, True),
            ("non-default-with-ids", False, ["break-default"], False),
        ],
    )
    def test_mysql_use_index_hint_behavior(
        self, monkeypatch, scenario, attach_tags, ids_value, expect_hint
    ):
        """
        Ensure the hint is applied for the UI-default behavior and not for simple deviations.
        """
        key = "artifact-for-default-query"
        self._db.store_artifact(
            self._db_session,
            key=key,
            artifact=self._generate_artifact(key, project=self.project),
            project=self.project,
        )

        hint_called = {"value": False}
        original_with_hint = Query.with_hint

        def with_hint_spy(q, selectable, text, dialect_name=None):
            if "USE INDEX" in str(text):
                hint_called["value"] = True
            return original_with_hint(q, selectable, text, dialect_name=dialect_name)

        monkeypatch.setattr(Query, "with_hint", with_hint_spy, raising=True)

        kwargs = {"project": self.project, **self._ui_defaults()}
        if attach_tags:
            kwargs["attach_tags"] = True
        if ids_value is not None:
            kwargs["ids"] = ids_value

        _ = self._db._find_artifacts(self._db_session, **kwargs)

        if expect_hint:
            assert hint_called["value"], f"{scenario}: expected USE INDEX hint"
        else:
            assert not hint_called["value"], (
                f"{scenario}: did not expect USE INDEX hint"
            )

    @staticmethod
    def _ui_defaults():
        return {
            "tag": mlrun.common.constants.RESERVED_TAG_NAME_LATEST,
            "best_iteration": True,
            "limit": 1001,
        }

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
            kind=mlrun.common.schemas.ObjectKind.project,
        )
        self._db.create_project(self._db_session, project)
