import deepdiff
import pandas

import mlrun
import mlrun.artifacts
import tests.integration.sdk_api.base


class TestArtifactTags(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_list_artifact_tags(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project.save_to_db()
        artifact_tags = mlrun.get_run_db().list_artifact_tags(project_name)
        assert artifact_tags == []
        key = "some-key"
        data_frame = pandas.DataFrame({"x": [1, 2]})
        artifact = mlrun.artifacts.dataset.DatasetArtifact(key, data_frame)
        uid = "some-uid"
        uid_2 = "some-uid-2"
        tag = "some-tag"
        tag_2 = "some-tag-2"
        mlrun.get_run_db().store_artifact(
            key, artifact.to_dict(), uid, tag=tag, project=project_name
        )
        mlrun.get_run_db().store_artifact(
            key, artifact.to_dict(), uid_2, tag=tag_2, project=project_name
        )
        artifact_tags = mlrun.get_run_db().list_artifact_tags(project_name)
        assert deepdiff.DeepDiff(artifact_tags, [tag, tag_2], ignore_order=True,) == {}
