# Copyright 2018 Iguazio
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
import pytest

import mlrun
import mlrun.feature_store as fs
import tests.integration.sdk_api.base
from mlrun.datastore import StreamSource
from mlrun.features import Entity


class TestFeatureStore(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_deploy_ingestion_service_without_preview(self):
        name = "deploy-without-preview"
        stream_path = f"/{self.project_name}/FeatureStore/{name}/v3ioStream"

        v3io_source = StreamSource(
            path=f"v3io:///projects{stream_path}", key_field="ticker", time_field="time"
        )
        fset = fs.FeatureSet(name, timestamp_key="time", entities=[Entity("ticker")])

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            fs.deploy_ingestion_service(
                featureset=fset,
                source=v3io_source,
            )

    def test_remove_labels_from_feature_set(self):
        db = mlrun.get_run_db()
        mlrun.get_or_create_project(self.project_name, "./")
        fset = fs.FeatureSet(
            "feature-set-test", timestamp_key="time", entities=[Entity("ticker")]
        )
        labels = {"label1": "value1", "label2": "value2"}
        fset.metadata.labels = labels

        db.store_feature_set(fset, project=self.project_name)
        feature_sets = db.list_feature_sets(project=self.project_name)
        assert len(feature_sets) == 1, "bad number of feature sets"
        assert (
            feature_sets[0].metadata.labels == labels
        ), "labels were not set correctly"

        fset.metadata.labels = {}
        db.store_feature_set(fset.to_dict(), project=self.project_name)
        feature_sets = db.list_feature_sets(project=self.project_name)
        for feature_set in feature_sets:
            if feature_set.metadata.tag == "latest":
                assert (
                    feature_set.metadata.labels is None
                ), "labels were not removed correctly"
                break
        else:
            assert False, "latest feature set not found"
