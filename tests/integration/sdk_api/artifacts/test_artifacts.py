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
import os
import pathlib
import shutil
import unittest.mock

import pandas

import mlrun
import mlrun.artifacts
import mlrun.common.schemas
import tests.integration.sdk_api.base
from tests import conftest

results_dir = (pathlib.Path(conftest.results) / "artifacts").absolute()


class TestArtifacts(tests.integration.sdk_api.base.TestMLRunIntegration):
    extra_env = {"MLRUN_HTTPDB__REAL_PATH": "/"}

    def test_artifacts(self):
        db = mlrun.get_run_db()
        prj, tree, key, body = "p9", "t19", "k802", "tomato"
        mlrun.get_or_create_project(prj, "./", allow_cross_project=True)
        artifact = mlrun.artifacts.Artifact(key, body, target_path="/a.txt")

        db.store_artifact(key, artifact, tree=tree, project=prj)
        db.store_artifact(key, artifact, tree=tree, project=prj, iter=42)
        artifacts = db.list_artifacts(project=prj, tag="*", tree=tree)
        assert len(artifacts) == 2, "bad number of artifacts"
        assert artifacts.to_objects()[0].key == key, "not a valid artifact object"
        assert artifacts.dataitems()[0].url, "not a valid artifact dataitem"

        artifacts = db.list_artifacts(project=prj, tag="*", iter=0)
        assert len(artifacts) == 1, "bad number of artifacts"

        # Only 1 will be returned since it's only looking for iter 0
        artifacts = db.list_artifacts(project=prj, tag="*", best_iteration=True)
        assert len(artifacts) == 1, "bad number of artifacts"

        db.del_artifacts(project=prj, tag="*")
        artifacts = db.list_artifacts(project=prj, tag="*")
        assert len(artifacts) == 0, "bad number of artifacts after del"

    def test_list_artifacts_filter_by_kind(self):
        prj, tree, key, body = "p9", "t19", "k802", "tomato"
        mlrun.get_or_create_project(prj, context="./", allow_cross_project=True)
        model_artifact = mlrun.artifacts.model.ModelArtifact(
            key, body, target_path="/a.txt"
        )

        data = {"col1": [1, 2], "col2": [3, 4]}
        data_frame = pandas.DataFrame(data=data)
        dataset_artifact = mlrun.artifacts.dataset.DatasetArtifact(
            key, df=data_frame, format="parquet", target_path="/b.pq"
        )

        db = mlrun.get_run_db()
        db.store_artifact(key, model_artifact, tree=f"model_{tree}", project=prj)
        db.store_artifact(
            key, dataset_artifact, tree=f"ds_{tree}", project=prj, iter=42
        )

        artifacts = db.list_artifacts(project=prj)
        assert len(artifacts) == 2, "bad number of artifacts"

        artifacts = db.list_artifacts(project=prj, kind="model")
        assert len(artifacts) == 1, "bad number of model artifacts"

        artifacts = db.list_artifacts(
            project=prj, category=mlrun.common.schemas.ArtifactCategories.dataset
        )
        assert len(artifacts) == 1, "bad number of dataset artifacts"

    def test_export_import(self):
        project = mlrun.new_project("log-mod")
        target_project = mlrun.new_project("log-mod2")
        for mode in [False, True]:
            mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = mode

            model = project.log_model(
                "mymod",
                body=b"123",
                model_file="model.pkl",
                extra_data={"kk": b"456"},
                artifact_path=results_dir,
            )

            for suffix in ["yaml", "json", "zip"]:
                # export the artifact to a file
                model.export(f"{results_dir}/a.{suffix}")

                new_key = f"mod-{suffix}"

                # import and log the artifact to the new project
                artifact = target_project.import_artifact(
                    f"{results_dir}/a.{suffix}",
                    new_key=new_key,
                    artifact_path=results_dir,
                )
                assert artifact.kind == "model"
                assert artifact.metadata.key == new_key
                assert artifact.spec.db_key == new_key
                assert artifact.metadata.project == "log-mod2"
                temp_path, model_spec, extra_dataitems = mlrun.artifacts.get_model(
                    artifact.uri
                )
                with open(temp_path, "rb") as fp:
                    data = fp.read()
                assert data == b"123"
                assert extra_dataitems["kk"].get() == b"456"

    def test_import_remote_zip(self):
        project = mlrun.new_project("log-mod")
        target_project = mlrun.new_project("log-mod2")
        model = project.log_model(
            "mymod",
            body=b"123",
            model_file="model.pkl",
            extra_data={"kk": b"456"},
            artifact_path=results_dir,
        )

        artifact_url = f"{results_dir}/a.zip"
        model.export(artifact_url)

        # mock downloading the artifact from s3 by copying it locally to a temp path
        mlrun.datastore.base.DataStore.download = unittest.mock.MagicMock(
            side_effect=shutil.copyfile
        )
        artifact = target_project.import_artifact(
            f"s3://ֿ{results_dir}/a.zip",
            "mod-zip",
            artifact_path=results_dir,
        )

        temp_local_path = mlrun.datastore.base.DataStore.download.call_args[0][1]
        assert artifact.metadata.project == "log-mod2"
        # verify that the original artifact was not deleted
        assert os.path.exists(artifact_url)
        # verify that the temp path was deleted after the import
        assert not os.path.exists(temp_local_path)
