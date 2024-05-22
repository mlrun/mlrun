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

IMAGE_NAME_ENRICH_REGISTRY_PREFIX = "."  # prefix for image name to enrich with registry
MLRUN_CREATED_LABEL = "mlrun-created"
MLRUN_MODEL_CONF = "model-conf"
MLRUN_SERVING_SPEC_MOUNT_PATH = f"/tmp/mlrun/{MLRUN_MODEL_CONF}"
MLRUN_SERVING_SPEC_FILENAME = "serving_spec.json"
MLRUN_SERVING_SPEC_PATH = (
    f"{MLRUN_SERVING_SPEC_MOUNT_PATH}/{MLRUN_SERVING_SPEC_FILENAME}"
)
MYSQL_MEDIUMBLOB_SIZE_BYTES = 16 * 1024 * 1024
MLRUN_KEY = "mlrun/"


class MlrunInternalLabels:
    dask_cluster_name = "dask.org/cluster-name"
    dask_component = "dask.org/component"
    host = "host"
    job_type = "job-type"
    kind = "kind"
    mlrun_auth_key = "mlrun-auth-key"
    nuclio_project_name = "nuclio.io/project-name"
    mlrun_class = f"{MLRUN_KEY}class"
    client_python_version = f"{MLRUN_KEY}client_python_version"
    client_version = f"{MLRUN_KEY}client_version"
    function = f"{MLRUN_KEY}function"
    job = f"{MLRUN_KEY}job"
    name = f"{MLRUN_KEY}name"
    mlrun_owner = f"{MLRUN_KEY}owner"
    owner_domain = f"{MLRUN_KEY}owner_domain"
    project = f"{MLRUN_KEY}project"
    runner_pod = f"{MLRUN_KEY}runner-pod"
    schedule_name = f"{MLRUN_KEY}schedule-name"
    scrape_metrics = f"{MLRUN_KEY}scrape-metrics"
    tag = f"{MLRUN_KEY}tag"
    uid = f"{MLRUN_KEY}uid"
    username = f"{MLRUN_KEY}username"
    username_domain = f"{MLRUN_KEY}username_domain"
    owner = "owner"
    resource_name = "resource_name"
    v3io_user = "v3io_user"
    workflow = "workflow"

    @classmethod
    def all(cls):
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]
