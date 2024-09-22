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


import datetime
from typing import Optional, Union

import mlrun.alerts
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.lists

from ..config import config
from ..utils import logger
from .base import RunDBInterface


class NopDB(RunDBInterface):
    def __init__(self, url=None, *args, **kwargs):
        self.url = url

    def __getattribute__(self, attr):
        def nop(*args, **kwargs):
            env_var_message = (
                "MLRUN_DBPATH is misconfigured. Set this environment variable to the URL of the API "
                "server in order to connect"
            )
            if config.httpdb.nop_db.raise_error:
                raise mlrun.errors.MLRunBadRequestError(env_var_message)

            if config.httpdb.nop_db.verbose:
                logger.warning(
                    "Could not detect path to API server, not connected to API server!"
                )
                logger.warning(env_var_message)

            return

        # ignore __class__ because __getattribute__ overrides the parent class's method and it spams logs
        if attr in ["connect", "__class__"]:
            return super().__getattribute__(attr)
        else:
            nop()
            return super().__getattribute__(attr)

    def connect(self, secrets=None):
        pass

    def store_log(self, uid, project="", body=None, append=False):
        pass

    def get_log(self, uid, project="", offset=0, size=0):
        pass

    def store_run(self, struct, uid, project="", iter=0):
        pass

    def update_run(self, updates: dict, uid, project="", iter=0):
        pass

    def abort_run(self, uid, project="", iter=0, timeout=45, status_text=""):
        pass

    def list_runtime_resources(
        self,
        project: Optional[str] = None,
        label_selector: Optional[str] = None,
        kind: Optional[str] = None,
        object_id: Optional[str] = None,
        group_by: Optional[
            mlrun.common.schemas.ListRuntimeResourcesGroupByField
        ] = None,
    ) -> Union[
        mlrun.common.schemas.RuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByJobRuntimeResourcesOutput,
        mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    ]:
        return []

    def read_run(
        self,
        uid,
        project="",
        iter=0,
        format_: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ):
        pass

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, list[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, list[str]]] = None,
        state: Optional[
            mlrun.common.runtimes.constants.RunStates
        ] = None,  # Backward compatibility
        states: Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
        start_time_from: datetime.datetime = None,
        start_time_to: datetime.datetime = None,
        last_update_time_from: datetime.datetime = None,
        last_update_time_to: datetime.datetime = None,
        partition_by: Union[mlrun.common.schemas.RunPartitionByField, str] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        with_notifications: bool = False,
    ):
        return mlrun.lists.RunList()

    def del_run(self, uid, project="", iter=0):
        pass

    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):
        pass

    def store_artifact(
        self, key, artifact, uid=None, iter=None, tag="", project="", tree=None
    ):
        pass

    def read_artifact(
        self,
        key,
        tag="",
        iter=None,
        project="",
        tree=None,
        uid=None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        pass

    def list_artifacts(
        self,
        name="",
        project="",
        tag="",
        labels=None,
        since=None,
        until=None,
        iter: int = None,
        best_iteration: bool = False,
        kind: str = None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
        tree: str = None,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        limit: int = None,
    ):
        return mlrun.lists.ArtifactList()

    def del_artifact(
        self,
        key,
        tag="",
        project="",
        tree=None,
        uid=None,
        deletion_strategy: mlrun.common.schemas.artifact.ArtifactsDeletionStrategies = (
            mlrun.common.schemas.artifact.ArtifactsDeletionStrategies.metadata_only
        ),
        secrets: dict = None,
        iter=None,
    ):
        pass

    def del_artifacts(self, name="", project="", tag="", labels=None):
        pass

    def store_function(self, function, name, project="", tag="", versioned=False):
        pass

    def get_function(self, name, project="", tag="", hash_key=""):
        pass

    def delete_function(self, name: str, project: str = ""):
        pass

    def list_functions(
        self, name=None, project="", tag="", labels=None, since=None, until=None
    ):
        return []

    def tag_objects(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
        replace: bool = False,
    ):
        pass

    def delete_objects_tag(
        self, project: str, tag_name: str, tag_objects: mlrun.common.schemas.TagObjects
    ):
        pass

    def tag_artifacts(
        self, artifacts, project: str, tag_name: str, replace: bool = False
    ):
        pass

    def delete_artifacts_tags(self, artifacts, project: str, tag_name: str):
        pass

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    def store_project(
        self, name: str, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.Project:
        pass

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.common.schemas.Project:
        pass

    def create_project(
        self, project: mlrun.common.schemas.Project
    ) -> mlrun.common.schemas.Project:
        pass

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.name_only,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    def get_project(self, name: str) -> mlrun.common.schemas.Project:
        pass

    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        pass

    def create_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        project="",
        versioned=True,
    ) -> dict:
        pass

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        pass

    def list_features_v2(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutputV2:
        pass

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: list[str] = None
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

    def list_entities_v2(
        self, project: str, name: str = None, tag: str = None, labels: list[str] = None
    ) -> mlrun.common.schemas.EntitiesOutputV2:
        pass

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: list[str] = None,
        features: list[str] = None,
        labels: list[str] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        format_: Union[
            str, mlrun.common.formatters.FeatureSetFormat
        ] = mlrun.common.formatters.FeatureSetFormat.full,
    ) -> list[dict]:
        pass

    def store_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        pass

    def patch_feature_set(
        self,
        name,
        feature_set: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    def delete_feature_set(self, name, project="", tag=None, uid=None):
        pass

    def create_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        pass

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: list[str] = None,
        partition_by: Union[
            mlrun.common.schemas.FeatureStorePartitionByField, str
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Union[mlrun.common.schemas.SortField, str] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
    ) -> list[dict]:
        pass

    def store_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        name=None,
        project="",
        tag=None,
        uid=None,
        versioned=True,
    ):
        pass

    def patch_feature_vector(
        self,
        name,
        feature_vector_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode: Union[
            str, mlrun.common.schemas.PatchMode
        ] = mlrun.common.schemas.PatchMode.replace,
    ):
        pass

    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        pass

    def get_pipeline(
        self,
        run_id: str,
        namespace: str = None,
        timeout: int = 30,
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.summary,
        project: str = None,
    ):
        pass

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.common.formatters.PipelineFormat
        ] = mlrun.common.formatters.PipelineFormat.metadata_only,
        page_size: int = None,
    ) -> mlrun.common.schemas.PipelinesOutput:
        return mlrun.common.schemas.PipelinesOutput(runs=[], total_size=0)

    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        pass

    def list_project_secrets(
        self,
        project: str,
        token: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: list[str] = None,
    ) -> mlrun.common.schemas.SecretsData:
        pass

    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        pass

    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: list[str] = None,
    ):
        pass

    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        pass

    def delete_model_endpoint(self, project: str, endpoint_id: str):
        pass

    def list_model_endpoints(
        self,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: list[str] = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: Optional[list[str]] = None,
    ):
        pass

    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        features: bool = False,
    ):
        pass

    def patch_model_endpoint(self, project: str, endpoint_id: str, attributes: dict):
        pass

    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        pass

    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        pass

    def list_hub_sources(
        self,
        item_name: Optional[str] = None,
        tag: Optional[str] = None,
        version: Optional[str] = None,
    ):
        pass

    def get_hub_source(self, source_name: str):
        pass

    def delete_hub_source(self, source_name: str):
        pass

    def get_hub_catalog(
        self,
        source_name: str,
        channel: str = None,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        pass

    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        channel: str = "development",
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        pass

    def store_api_gateway(
        self,
        api_gateway: Union[
            mlrun.common.schemas.APIGateway,
            mlrun.runtimes.nuclio.api_gateway.APIGateway,
        ],
        project: str = None,
    ) -> mlrun.common.schemas.APIGateway:
        pass

    def list_api_gateways(self, project=None):
        pass

    def get_api_gateway(self, name, project=None):
        pass

    def delete_api_gateway(self, name, project=None):
        pass

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        pass

    def remote_builder(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        with_mlrun: bool,
        mlrun_version_specifier: Optional[str] = None,
        skip_deployed: bool = False,
        builder_env: Optional[dict] = None,
        force_build: bool = False,
    ):
        pass

    def deploy_nuclio_function(
        self,
        func: "mlrun.runtimes.RemoteRuntime",
        builder_env: Optional[dict] = None,
    ):
        pass

    def get_builder_status(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        pass

    def get_nuclio_deploy_status(
        self,
        func: "mlrun.runtimes.RemoteRuntime",
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        pass

    def set_run_notifications(
        self,
        project: str,
        runs: list[mlrun.model.RunObject],
        notifications: list[mlrun.model.Notification],
    ):
        pass

    def store_run_notifications(
        self,
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
        mask_params: bool = True,
    ):
        pass

    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str,
        mask_params: bool = True,
    ):
        pass

    def get_log_size(self, uid, project=""):
        pass

    def watch_log(self, uid, project="", watch=True, offset=0):
        pass

    def get_datastore_profile(
        self, name: str, project: str
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        pass

    def delete_datastore_profile(self, name: str, project: str):
        pass

    def list_datastore_profiles(
        self, project: str
    ) -> list[mlrun.common.schemas.DatastoreProfile]:
        pass

    def store_datastore_profile(
        self, profile: mlrun.common.schemas.DatastoreProfile, project: str
    ):
        pass

    def function_status(self, project, name, kind, selector):
        pass

    def start_function(
        self, func_url: str = None, function: "mlrun.runtimes.BaseRuntime" = None
    ):
        pass

    def submit_workflow(
        self,
        project: str,
        name: str,
        workflow_spec: Union[
            "mlrun.projects.pipelines.WorkflowSpec",
            "mlrun.common.schemas.WorkflowSpec",
            dict,
        ],
        arguments: Optional[dict] = None,
        artifact_path: Optional[str] = None,
        source: Optional[str] = None,
        run_name: Optional[str] = None,
        namespace: Optional[str] = None,
        notifications: list["mlrun.model.Notification"] = None,
    ) -> "mlrun.common.schemas.WorkflowResponse":
        pass

    def update_model_monitoring_controller(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
    ):
        pass

    def enable_model_monitoring(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        rebuild_images: bool = False,
        fetch_credentials_from_sys_config: bool = False,
    ) -> None:
        pass

    def disable_model_monitoring(
        self,
        project: str,
        delete_resources: bool = True,
        delete_stream_function: bool = False,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: list[str] = None,
    ) -> bool:
        pass

    def delete_model_monitoring_function(
        self, project: str, functions: list[str]
    ) -> bool:
        pass

    def deploy_histogram_data_drift_app(
        self, project: str, image: str = "mlrun/mlrun"
    ) -> None:
        pass

    def set_model_monitoring_credentials(
        self,
        project: str,
        credentials: dict[str, str],
        replace_creds: bool,
    ) -> None:
        pass

    def generate_event(
        self, name: str, event_data: Union[dict, mlrun.common.schemas.Event], project=""
    ):
        pass

    def store_alert_config(
        self,
        alert_name: str,
        alert_data: Union[dict, mlrun.alerts.alert.AlertConfig],
        project="",
    ):
        pass

    def get_alert_config(self, alert_name: str, project=""):
        pass

    def list_alerts_configs(self, project=""):
        pass

    def delete_alert_config(self, alert_name: str, project=""):
        pass

    def reset_alert_config(self, alert_name: str, project=""):
        pass

    def get_alert_template(self, template_name: str):
        pass

    def list_alert_templates(self):
        pass
