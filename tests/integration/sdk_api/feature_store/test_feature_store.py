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
import mlrun.feature_store as fstore
import tests.integration.sdk_api.base
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
