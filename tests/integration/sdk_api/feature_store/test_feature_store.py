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
import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fstore
import tests.integration.sdk_api.base
from mlrun.data_types import InferOptions
from mlrun.datastore import StreamSource
from mlrun.features import Entity


class TestFeatureStore(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_deploy_ingestion_service_without_preview(self):
        name = "deploy-without-preview"
        stream_path = f"/{self.project_name}/FeatureStore/{name}/v3ioStream"

        v3io_source = StreamSource(
            path=f"v3io:///projects{stream_path}",
            key_field="ticker",
        )
        fset = fstore.FeatureSet(
            name, timestamp_key="time", entities=[Entity("ticker")]
        )

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            fstore.deploy_ingestion_service(
                featureset=fset,
                source=v3io_source,
            )

    def test_get_online_features_after_ingest_without_inference(self):
        feature_set = fstore.FeatureSet(
            "my-fset",
            entities=[
                fstore.Entity("fn0"),
                fstore.Entity(
                    "fn1",
                    value_type=mlrun.data_types.data_types.ValueType.STRING,
                ),
            ],
        )
        # feature_set.set_targets(
        #     targets=[mlrun.datastore.NoSqlTarget()], with_defaults=False
        # )
        # feature_set.save()

        df = pd.DataFrame(
            {
                "fn0": [1, 2, 3, 4],
                "fn1": [1, 2, 3, 4],
                "fn2": [1, 1, 1, 1],
                "fn3": [2, 2, 2, 2],
            }
        )

        fstore.ingest(feature_set, df, infer_options=InferOptions.Null)

        features = ["my-fset.*"]
        vector = fstore.FeatureVector("my-vector", features)
        vector.save()

        sv = fstore.get_online_feature_service(
            f"store://feature-vectors/{self.project_name}/my-vector:latest"
        )
        try:
            with pytest.raises(mlrun.errors.MLRunRuntimeError) as err:
                sv.get([{"fn0": "1", "fn1": "1"}])
        finally:
            sv.close()
        assert err.value == "No features found for feature vector 'my-vector'"
