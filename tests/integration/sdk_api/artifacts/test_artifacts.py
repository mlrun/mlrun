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
