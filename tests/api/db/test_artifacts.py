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
import os
import tempfile

import deepdiff
import numpy
import pandas
import pytest
from sqlalchemy.orm import Session

import mlrun.api.db.sqldb.models
import mlrun.api.initial_data
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
from mlrun.api.db.base import DBInterface
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact
from mlrun.common.schemas.artifact import ArtifactCategories


def test_list_artifact_name_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_name_2 = "artifact_name_2"
    tree = "artifact_tree"
    artifact_1 = _generate_artifact(artifact_name_1, tree=tree)
    artifact_2 = _generate_artifact(artifact_name_2, tree=tree)

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


def test_list_artifact_iter_parameter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_name_2 = "artifact_name_2"
    tree = "artifact_tree"
    artifact_1 = _generate_artifact(artifact_name_1, tree=tree)
    artifact_2 = _generate_artifact(artifact_name_2, tree=tree)

    # Use iters with multiple digits, to make sure filtering them via regex works
    test_iters = [0, 5, 9, 42, 219, 2102]
    for iter in test_iters:
        artifact_1["iter"] = artifact_2["iter"] = iter
        db.store_artifact(db_session, artifact_name_1, artifact_1, iter=iter)
        db.store_artifact(db_session, artifact_name_2, artifact_2, iter=iter)

    # No filter on iter. All are expected
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == len(test_iters) * 2

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


def test_list_artifact_kind_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_kind_1 = ChartArtifact.kind
    artifact_name_2 = "artifact_name_2"
    artifact_kind_2 = PlotArtifact.kind
    tree = "artifact_tree"
    artifact_1 = _generate_artifact(artifact_name_1, kind=artifact_kind_1, tree=tree)
    artifact_2 = _generate_artifact(artifact_name_2, kind=artifact_kind_2, tree=tree)

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


def test_list_artifact_category_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_kind_1 = ChartArtifact.kind
    artifact_name_2 = "artifact_name_2"
    artifact_kind_2 = PlotArtifact.kind
    artifact_name_3 = "artifact_name_3"
    artifact_kind_3 = ModelArtifact.kind
    artifact_name_4 = "artifact_name_4"
    artifact_kind_4 = DatasetArtifact.kind
    tree = "artifact_tree"
    artifact_1 = _generate_artifact(artifact_name_1, kind=artifact_kind_1, tree=tree)
    artifact_2 = _generate_artifact(artifact_name_2, kind=artifact_kind_2, tree=tree)
    artifact_3 = _generate_artifact(artifact_name_3, kind=artifact_kind_3, tree=tree)
    artifact_4 = _generate_artifact(artifact_name_4, kind=artifact_kind_4, tree=tree)

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


def test_store_artifact_tagging(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact_key_1"
    artifact_1_tree = "artifact_tree"
    artifact_1_tag = "artifact_tag_1"
    artifact_1_body = _generate_artifact(artifact_1_key, tree=artifact_1_tree)
    artifact_1_kind = ChartArtifact.kind
    artifact_1_with_kind_tree = "artifact_tree_2"
    artifact_2_tag = "artifact_tag_2"
    artifact_1_with_kind_body = _generate_artifact(
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


def test_store_artifact_restoring_multiple_tags(db: DBInterface, db_session: Session):
    artifact_key = "artifact_key_1"
    artifact_1_tree = "artifact_tree_1"
    artifact_2_tree = "artifact_tree_2"
    artifact_1_body = _generate_artifact(artifact_key, tree=artifact_1_tree)
    artifact_2_body = _generate_artifact(artifact_key, tree=artifact_2_tree)
    artifact_1_tag = "artifact-tag-1"
    artifact_2_tag = "artifact-tag-2"

    # we use deepcopy to avoid changing the original dict
    db.store_artifact(
        db_session,
        artifact_key,
        copy.deepcopy(artifact_1_body),
        tag=artifact_1_tag,
    )
    db.store_artifact(
        db_session,
        artifact_key,
        copy.deepcopy(artifact_2_body),
        tag=artifact_2_tag,
    )
    artifacts = db.list_artifacts(db_session, artifact_key, tag="*")
    assert len(artifacts) == 3  # also latest is returned

    # ids are auto generated using this util function
    expected_uids = [
        mlrun.utils.fill_artifact_object_hash(artifact_body, "uid")
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


def test_read_artifact_tag_resolution(db: DBInterface, db_session: Session):
    """
    We had a bug in which when we got a tag filter for read/list artifact, we were transforming this tag to list of
    possible uids which is wrong, since a different artifact might have this uid as well, and we will return it,
    although it's not really tag with the given tag
    """
    artifact_1_key = "artifact_key_1"
    artifact_2_key = "artifact_key_2"
    artifact_tree = "artifact_uid_1"
    artifact_1_body = _generate_artifact(artifact_1_key, tree=artifact_tree)
    artifact_2_body = _generate_artifact(artifact_2_key, tree=artifact_tree)
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


def test_delete_artifacts_tag_filter(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact_key_1"
    artifact_2_key = "artifact_key_2"
    artifact_1_tree = "artifact_tree_1"
    artifact_2_tree = "artifact_tree_2"
    artifact_1_body = _generate_artifact(artifact_1_key, tree=artifact_1_tree)
    artifact_2_body = _generate_artifact(artifact_2_key, tree=artifact_2_tree)
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


def test_delete_artifact_tag_filter(db: DBInterface, db_session: Session):
    project = "artifact_project"
    artifact_1_key = "artifact_key_1"
    artifact_2_key = "artifact_key_2"
    artifact_1_tree = "artifact_tree_1"
    artifact_2_tree = "artifact_tree_2"
    artifact_1_body = _generate_artifact(artifact_1_key, tree=artifact_1_tree)
    artifact_2_body = _generate_artifact(artifact_2_key, tree=artifact_2_tree)
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
    db.del_artifact(db_session, artifact_2_key, tag=artifact_2_tag_2, project=project)
    artifacts = db.list_artifacts(db_session, project=project, name=artifact_2_key)
    assert len(artifacts) == 0

    # Assert all tags were deleted
    tags = db.list_artifact_tags(db_session, project)
    assert len(tags) == 0


def test_list_artifacts_exact_name_match(db: DBInterface, db_session: Session):
    artifact_1_key = "pre_artifact_key_suffix"
    artifact_2_key = "pre-artifact-key-suffix"
    artifact_1_tree = "artifact_tree_1"
    artifact_2_tree = "artifact_tree_2"
    artifact_1_body = _generate_artifact(artifact_1_key, tree=artifact_1_tree)
    artifact_2_body = _generate_artifact(artifact_2_key, tree=artifact_2_tree)

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


def _generate_artifact_with_iterations(
    db, db_session, key, tree, num_iters, best_iter, kind, project=""
):
    # using reversed so the link artifact will be created last, after all the iterations
    # are already stored
    for iter in reversed(range(num_iters)):
        artifact_body = _generate_artifact(
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
        )


def test_list_artifacts_best_iter_with_tagged_iteration(
    db: DBInterface, db_session: Session
):
    artifact_key_1 = "artifact-1"
    artifact_key_2 = "artifact-2"
    artifact_tree_1 = "tree-1"
    artifact_tree_2 = "tree-2"
    num_iters = 3
    best_iter = 2
    project = "project1"
    tag = "mytag1"

    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_key_1,
        artifact_tree_1,
        num_iters,
        best_iter,
        ArtifactCategories.model,
        project=project,
    )

    _generate_artifact_with_iterations(
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
    db.append_tag_to_artifacts(db_session, project, tag, [identifier_1, identifier_2])
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


def test_list_artifacts_best_iter(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact-1"
    artifact_1_tree = "tree-1"
    artifact_2_key = "artifact-2"
    artifact_2_tree = "tree-2"
    artifact_no_link_key = "single-artifact"
    artifact_no_link_tree = "tree-3"

    num_iters = 5
    best_iter_1 = 2
    best_iter_2 = 4
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_1_key,
        artifact_1_tree,
        num_iters,
        best_iter_1,
        ArtifactCategories.model,
    )
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_2_key,
        artifact_2_tree,
        num_iters,
        best_iter_2,
        ArtifactCategories.dataset,
    )

    # Add non-hyper-param artifact. Single object with iter 0, not pointing at anything
    artifact_body = _generate_artifact(artifact_no_link_key, artifact_no_link_tree)
    artifact_body["spec"]["iter"] = 0
    db.store_artifact(db_session, artifact_no_link_key, artifact_body, iter=0)

    results = db.list_artifacts(db_session, name="~artifact")
    # we don't store link artifacts in the DB, so we expect 2 * num_iters - 1, plus a regular artifact
    assert len(results) == (num_iters - 1) * 2 + 1

    results = db.list_artifacts(db_session, name=artifact_1_key, best_iteration=True)
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


def test_list_artifacts_best_iteration(db: DBInterface, db_session: Session):
    artifact_key = "artifact-1"
    artifact_1_tree = "tree-1"
    artifact_2_tree = "tree-2"
    artifact_3_tree = "tree-3"

    num_iters = 5
    best_iter_1 = 2
    best_iter_2 = 4
    best_iter_3 = 2
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_key,
        artifact_1_tree,
        num_iters,
        best_iter_1,
        ArtifactCategories.model,
    )
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_key,
        artifact_2_tree,
        num_iters,
        best_iter_2,
        ArtifactCategories.model,
    )
    _generate_artifact_with_iterations(
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
            assert result["metadata"]["tag"] == "latest"


def test_data_migration_fix_legacy_datasets_large_previews(
    data_migration_db: DBInterface,
    db_session: Session,
):
    artifact_with_valid_preview_key = "artifact-with-valid-preview-key"
    artifact_with_valid_preview_uid = "artifact-with-valid-preview-uid"
    artifact_with_valid_preview = mlrun.artifacts.dataset.LegacyDatasetArtifact(
        artifact_with_valid_preview_key,
        df=pandas.DataFrame(
            [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
        ),
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_valid_preview_key,
        artifact_with_valid_preview.to_dict(),
        artifact_with_valid_preview_uid,
    )

    artifact_with_invalid_preview_key = "artifact-with-invalid-preview-key"
    artifact_with_invalid_preview_uid = "artifact-with-invalid-preview-uid"
    artifact_with_invalid_preview = mlrun.artifacts.dataset.LegacyDatasetArtifact(
        artifact_with_invalid_preview_key,
        df=pandas.DataFrame(
            numpy.random.randint(
                0, 10, size=(10, mlrun.artifacts.dataset.max_preview_columns * 3)
            )
        ),
        ignore_preview_limits=True,
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_invalid_preview_key,
        artifact_with_invalid_preview.to_dict(),
        artifact_with_invalid_preview_uid,
    )

    # perform the migration
    mlrun.api.initial_data._fix_datasets_large_previews(data_migration_db, db_session)

    artifact_with_valid_preview_after_migration = data_migration_db.read_artifact_v1(
        db_session, artifact_with_valid_preview_key, artifact_with_valid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_valid_preview_after_migration,
            artifact_with_valid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=["root['updated']", "root['tag']", "root['db_key']"],
        )
        == {}
    )

    artifact_with_invalid_preview_after_migration = data_migration_db.read_artifact_v1(
        db_session, artifact_with_invalid_preview_key, artifact_with_invalid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_invalid_preview_after_migration,
            artifact_with_invalid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['updated']",
                "root['header']",
                "root['stats']",
                "root['schema']",
                "root['preview']",
                "root['tag']",
                "root['db_key']",
            ],
        )
        == {}
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["header"])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["stats"])
        == mlrun.artifacts.dataset.max_preview_columns - 1
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["preview"][0])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["schema"]["fields"])
        == mlrun.artifacts.dataset.max_preview_columns + 1
    )


def test_data_migration_fix_datasets_large_previews(
    data_migration_db: DBInterface,
    db_session: Session,
):
    artifact_with_valid_preview_key = "artifact-with-valid-preview-key"
    artifact_with_valid_preview_uid = "artifact-with-valid-preview-uid"
    artifact_with_valid_preview = mlrun.artifacts.dataset.DatasetArtifact(
        artifact_with_valid_preview_key,
        df=pandas.DataFrame(
            [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
        ),
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_valid_preview_key,
        artifact_with_valid_preview.to_dict(),
        artifact_with_valid_preview_uid,
    )

    artifact_with_invalid_preview_key = "artifact-with-invalid-preview-key"
    artifact_with_invalid_preview_uid = "artifact-with-invalid-preview-uid"
    artifact_with_invalid_preview = mlrun.artifacts.dataset.DatasetArtifact(
        artifact_with_invalid_preview_key,
        df=pandas.DataFrame(
            numpy.random.randint(
                0, 10, size=(10, mlrun.artifacts.dataset.max_preview_columns * 3)
            )
        ),
        ignore_preview_limits=True,
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_invalid_preview_key,
        artifact_with_invalid_preview.to_dict(),
        artifact_with_invalid_preview_uid,
    )

    # perform the migration
    mlrun.api.initial_data._fix_datasets_large_previews(data_migration_db, db_session)

    artifact_with_valid_preview_after_migration = data_migration_db.read_artifact_v1(
        db_session, artifact_with_valid_preview_key, artifact_with_valid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_valid_preview_after_migration,
            artifact_with_valid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['metadata']['updated']",
                "root['metadata']['tag']",
                "root['spec']['db_key']",
            ],
        )
        == {}
    )

    artifact_with_invalid_preview_after_migration = data_migration_db.read_artifact_v1(
        db_session, artifact_with_invalid_preview_key, artifact_with_invalid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_invalid_preview_after_migration,
            artifact_with_invalid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['metadata']['updated']",
                "root['spec']['header']",
                "root['status']['stats']",
                "root['spec']['schema']",
                "root['status']['preview']",
                "root['metadata']['tag']",
                "root['spec']['db_key']",
            ],
        )
        == {}
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["spec"]["header"])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["status"]["stats"])
        == mlrun.artifacts.dataset.max_preview_columns - 1
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["status"]["preview"][0])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["spec"]["schema"]["fields"])
        == mlrun.artifacts.dataset.max_preview_columns + 1
    )


def test_migrate_artifacts_to_v2(db: DBInterface, db_session: Session):
    artifact_key = "artifact1"
    artifact_uid = "uid1"
    project = "project1"

    # create project
    db.create_project(
        db_session,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=project,
            ),
            spec=mlrun.common.schemas.ProjectSpec(description="some-description"),
        ),
    )

    # create an artifact in the old format
    artifact_body = _generate_artifact(artifact_key, artifact_uid, "artifact")
    artifact_body["metadata"]["key"] = artifact_key
    artifact_body["metadata"]["iter"] = 2
    artifact_body["metadata"]["project"] = project
    db.store_artifact_v1(
        db_session, artifact_key, artifact_body, artifact_uid, project=project
    )

    # create a legacy artifact in the old format
    legacy_artifact_key = "legacy-dataset-artifact1"
    legacy_artifact_uid = "legacy-uid1"
    legacy_artifact = {
        "key": legacy_artifact_key,
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
    )

    try:
        # change working directory to temp directory so the state file will be created there
        current_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # perform the migration
            mlrun.api.initial_data._migrate_artifacts_table_v2(db, db_session)
    finally:
        # change working directory back to original directory
        os.chdir(current_dir)

    # validate the migration succeeded
    query_all = db._query(
        db_session,
        mlrun.api.db.sqldb.models.ArtifactV2,
    )
    new_artifacts = query_all.all()
    assert len(new_artifacts) == 2

    for expected in [
        {
            "key": artifact_key,
            "uid": artifact_uid,
            "project": project,
            "iter": 2,
        },
        {
            "key": legacy_artifact_key,
            "uid": legacy_artifact_uid,
            "project": None,
            "iter": None,
        },
    ]:
        # TODO: remove this query once the v2 db layer methods are implemented. This is just a temporary workaround
        query = db._query(
            db_session,
            mlrun.api.db.sqldb.models.ArtifactV2,
            key=expected["key"],
        )
        artifact = query.one_or_none()
        assert artifact is not None
        assert artifact.key == expected["key"]
        assert artifact.producer_id == expected["uid"]
        assert artifact.project == expected["project"]
        assert artifact.iter == expected["iter"]

        artifact_dict = artifact.full_object
        assert len(artifact_dict) > 0
        assert artifact_dict["metadata"]["key"] == expected["key"]
        if expected["project"] is not None:
            assert artifact_dict["metadata"]["project"] == expected["project"]
        else:
            assert "project" not in artifact_dict["metadata"]
        # the uid should be the generated uid and not the original one
        assert artifact_dict["metadata"]["uid"] != expected["uid"]

        # validate the original artifact was deleted
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.read_artifact_v1(
                db_session, expected["key"], project=expected["project"]
            )


def _generate_artifact(name, uid=None, kind=None, tree=None):
    artifact = {
        "metadata": {"key": name},
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

    return artifact
