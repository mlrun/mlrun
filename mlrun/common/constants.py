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
import mlrun.common.types

IMAGE_NAME_ENRICH_REGISTRY_PREFIX = "."  # prefix for image name to enrich with registry
MLRUN_SERVING_CONF = "serving-conf"
MLRUN_SERVING_SPEC_MOUNT_PATH = f"/tmp/mlrun/{MLRUN_SERVING_CONF}"
MLRUN_SERVING_SPEC_FILENAME = "serving_spec.json"
MLRUN_SERVING_SPEC_PATH = (
    f"{MLRUN_SERVING_SPEC_MOUNT_PATH}/{MLRUN_SERVING_SPEC_FILENAME}"
)
MLRUN_FUNCTIONS_ANNOTATION = "mlrun/mlrun-functions"
MYSQL_MEDIUMBLOB_SIZE_BYTES = 16 * 1024 * 1024
MLRUN_LABEL_PREFIX = "mlrun/"
DASK_LABEL_PREFIX = "dask.org/"
NUCLIO_LABEL_PREFIX = "nuclio.io/"
RESERVED_TAG_NAME_LATEST = "latest"


class MLRunInternalLabels:
    ### dask
    dask_cluster_name = f"{DASK_LABEL_PREFIX}cluster-name"
    dask_component = f"{DASK_LABEL_PREFIX}component"

    ### spark
    spark_role = "spark-role"

    ### mpi
    mpi_job_name = "mpi-job-name"
    mpi_job_role = "mpi-job-role"
    mpi_role_type = "mpi_role_type"

    ### nuclio
    nuclio_project_name = f"{NUCLIO_LABEL_PREFIX}project-name"
    nuclio_function_name = f"{NUCLIO_LABEL_PREFIX}function-name"
    nuclio_class = f"{NUCLIO_LABEL_PREFIX}class"

    ### mlrun
    mlrun_auth_key = "mlrun-auth-key"
    mlrun_class = f"{MLRUN_LABEL_PREFIX}class"
    client_python_version = f"{MLRUN_LABEL_PREFIX}client_python_version"
    client_version = f"{MLRUN_LABEL_PREFIX}client_version"
    function = f"{MLRUN_LABEL_PREFIX}function"
    job = f"{MLRUN_LABEL_PREFIX}job"
    name = f"{MLRUN_LABEL_PREFIX}name"
    mlrun_owner = f"{MLRUN_LABEL_PREFIX}owner"
    owner_domain = f"{MLRUN_LABEL_PREFIX}owner_domain"
    project = f"{MLRUN_LABEL_PREFIX}project"
    runner_pod = f"{MLRUN_LABEL_PREFIX}runner-pod"
    schedule_name = f"{MLRUN_LABEL_PREFIX}schedule-name"
    scrape_metrics = f"{MLRUN_LABEL_PREFIX}scrape-metrics"
    tag = f"{MLRUN_LABEL_PREFIX}tag"
    uid = f"{MLRUN_LABEL_PREFIX}uid"
    username = f"{MLRUN_LABEL_PREFIX}username"
    username_domain = f"{MLRUN_LABEL_PREFIX}username_domain"
    task_name = f"{MLRUN_LABEL_PREFIX}task-name"
    resource_name = f"{MLRUN_LABEL_PREFIX}resource_name"
    created = f"{MLRUN_LABEL_PREFIX}created"
    producer_type = f"{MLRUN_LABEL_PREFIX}producer-type"
    app_name = f"{MLRUN_LABEL_PREFIX}app-name"
    endpoint_id = f"{MLRUN_LABEL_PREFIX}endpoint-id"
    endpoint_name = f"{MLRUN_LABEL_PREFIX}endpoint-name"
    host = "host"
    job_type = "job-type"
    kind = "kind"
    component = "component"
    mlrun_type = "mlrun__type"

    owner = "owner"
    v3io_user = "v3io_user"
    workflow = "workflow"
    feature_vector = "feature-vector"

    @classmethod
    def all(cls):
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]

    @staticmethod
    def default_run_labels_to_enrich():
        return [
            MLRunInternalLabels.owner,
            MLRunInternalLabels.v3io_user,
        ]


class DeployStatusTextKind(mlrun.common.types.StrEnum):
    logs = "logs"
    events = "events"
