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

import pytest

from mlrun.datastore.datastore_profile import (
    DatastoreProfileGCS,
    register_temporary_client_datastore_profile,
)
from tests.system.feature_store.spark_hadoop_test_base import (
    Deployment,
    SparkHadoopTestBase,
)


@pytest.mark.skipif(
    not SparkHadoopTestBase._get_env_from_file().get("GCS_BUCKET_NAME"),
    reason="GCS_BUCKET_NAME is not set",
)
# Marked as enterprise because of v3io mount and remote spark
@pytest.mark.enterprise
class TestFeatureStoreGcsSparkEngine(SparkHadoopTestBase):
    @classmethod
    def custom_setup_class(cls):
        cls.configure_namespace("gcs")
        cls.env = os.environ
        cls.configure_image_deployment(Deployment.Remote, "google-cloud-storage")

    def test_basic_remote_spark_ingest_ds_gcs(self):
        ds_profile = DatastoreProfileGCS(
            name=self.ds_profile_name,
            gcp_credentials=self.env["GCP_CREDENTIALS"],
        )
        register_temporary_client_datastore_profile(ds_profile)
        self.project.register_datastore_profile(ds_profile)

        bucket = self.env["GCS_BUCKET_NAME"]
        self.ds_upload_src(ds_profile, bucket)

        self.do_test(
            self.ds_src_path(ds_profile, bucket),
            self.ds_target_path(ds_profile, bucket),
        )
