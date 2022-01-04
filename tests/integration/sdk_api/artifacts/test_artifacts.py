import pandas

import mlrun
import mlrun.artifacts
import tests.integration.sdk_api.base


class TestArtifacts(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_artifacts(self):
        db = mlrun.get_run_db()
        prj, uid, key, body = "p9", "u19", "k802", "tomato"
        artifact = mlrun.artifacts.Artifact(key, body, target_path="a.txt")

        db.store_artifact(key, artifact, uid, project=prj)
        db.store_artifact(key, artifact, uid, project=prj, iter=42)
        artifacts = db.list_artifacts(project=prj, tag="*")
        assert len(artifacts) == 2, "bad number of artifacts"
        assert artifacts.objects()[0].key == key, "not a valid artifact object"
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
        prj, uid, key, body = "p9", "u19", "k802", "tomato"
        model_artifact = mlrun.artifacts.model.ModelArtifact(
            key, body, target_path="a.txt"
        )

        data = {"col1": [1, 2], "col2": [3, 4]}
        data_frame = pandas.DataFrame(data=data)
        dataset_artifact = mlrun.artifacts.dataset.DatasetArtifact(
            key, df=data_frame, format="parquet", target_path="b.pq"
        )

        db = mlrun.get_run_db()
        db.store_artifact(key, model_artifact, f"model_{uid}", project=prj)
        db.store_artifact(key, dataset_artifact, f"ds_{uid}", project=prj, iter=42)

        artifacts = db.list_artifacts(project=prj)
        assert len(artifacts) == 2, "bad number of artifacts"

        artifacts = db.list_artifacts(project=prj, kind="model")
        assert len(artifacts) == 1, "bad number of model artifacts"

        artifacts = db.list_artifacts(
            project=prj, category=mlrun.api.schemas.ArtifactCategories.dataset
        )
        assert len(artifacts) == 1, "bad number of dataset artifacts"

    def test_store_artifact_with_empty_dict(self):
        project_name = "prj"
        project = mlrun.new_project(project_name)
        project.save_to_db()
        key = "some-key"
        uid = "some-uid"
        tag = "some-tag"
        mlrun.get_run_db().store_artifact(key, {}, uid, tag=tag, project=project_name)
        artifact_tags = mlrun.get_run_db().list_artifact_tags(project_name)
        assert artifact_tags == [tag]

    def test_del_artifacts_with_empty_dict_stored(self):
        project_name = "prj"
        project = mlrun.new_project(project_name)
        project.save_to_db()
        key1 = "some-key1"
        uid1 = "some-uid1"
        tag1 = "some-tag1"
        mlrun.get_run_db().store_artifact(
            key1, {}, uid1, tag=tag1, project=project_name
        )

        key2 = "some-key2"
        uid2 = "some-uid2"
        tag2 = "some-tag2"
        data_frame = pandas.DataFrame({"x": [1, 2]})
        artifact = mlrun.artifacts.dataset.DatasetArtifact(key2, data_frame)
        mlrun.get_run_db().store_artifact(
            key2, artifact.to_dict(), uid2, tag=tag2, project=project_name
        )
        mlrun.get_run_db().del_artifacts(project=project_name)
        artifact_tags = mlrun.get_run_db().list_artifact_tags(project_name)
        assert artifact_tags == []
