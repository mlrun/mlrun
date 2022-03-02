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
