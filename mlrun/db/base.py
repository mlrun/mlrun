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
from abc import ABC, abstractmethod
from typing import Optional, Union

from deprecated import deprecated

import mlrun.alerts
import mlrun.common
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.model_monitoring


class RunDBError(Exception):
    pass


class RunDBInterface(ABC):
    kind = ""

    @abstractmethod
    def connect(self, secrets=None):
        return self

    @abstractmethod
    def store_log(self, uid, project="", body=None, append=False):
        pass

    @abstractmethod
    def get_log(self, uid, project="", offset=0, size=0):
        pass

    @abstractmethod
    def store_run(self, struct, uid, project="", iter=0):
        pass

    @abstractmethod
    def update_run(self, updates: dict, uid, project="", iter=0):
        pass

    @abstractmethod
    def abort_run(self, uid, project="", iter=0, timeout=45, status_text=""):
        pass

    @abstractmethod
    def read_run(
        self,
        uid: str,
        project: str = "",
        iter: int = 0,
        format_: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def del_run(self, uid, project="", iter=0):
        pass

    @abstractmethod
    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):
        pass

    @abstractmethod
    def store_artifact(
        self, key, artifact, uid=None, iter=None, tag="", project="", tree=None
    ):
        pass

    @abstractmethod
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

    @abstractmethod
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
        pass

    @abstractmethod
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

    @abstractmethod
    def del_artifacts(self, name="", project="", tag="", labels=None):
        pass

    @abstractmethod
    def store_function(self, function, name, project="", tag="", versioned=False):
        pass

    @abstractmethod
    def get_function(self, name, project="", tag="", hash_key=""):
        pass

    @abstractmethod
    def delete_function(self, name: str, project: str = ""):
        pass

    @abstractmethod
    def list_functions(
        self, name=None, project="", tag="", labels=None, since=None, until=None
    ):
        pass

    @abstractmethod
    def tag_objects(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
        replace: bool = False,
    ):
        pass

    @abstractmethod
    def delete_objects_tag(
        self,
        project: str,
        tag_name: str,
        tag_objects: mlrun.common.schemas.TagObjects,
    ):
        pass

    @abstractmethod
    def tag_artifacts(
        self,
        artifacts,
        project: str,
        tag_name: str,
        replace: bool = False,
    ):
        pass

    @abstractmethod
    def delete_artifacts_tags(
        self,
        artifacts,
        project: str,
        tag_name: str,
    ):
        pass

    @staticmethod
    def _resolve_artifacts_to_tag_objects(
        artifacts,
    ) -> mlrun.common.schemas.TagObjects:
        """
        :param artifacts: Can be a list of :py:class:`~mlrun.artifacts.Artifact` objects or
            dictionaries, or a single object.
        :return: :py:class:`~mlrun.common.schemas.TagObjects`
        """
        # to avoid circular imports we import here
        import mlrun.artifacts.base

        if not isinstance(artifacts, list):
            artifacts = [artifacts]

        artifact_identifiers = []
        for artifact in artifacts:
            artifact_obj = (
                artifact.to_dict()
                if isinstance(artifact, mlrun.artifacts.base.Artifact)
                else artifact
            )
            artifact_identifiers.append(
                mlrun.common.schemas.ArtifactIdentifier(
                    # we pass the db_key and not the key so the API will be able to find the artifact in the db
                    key=mlrun.utils.get_in_artifact(artifact_obj, "db_key"),
                    uid=mlrun.utils.get_in_artifact(artifact_obj, "uid"),
                    producer_id=mlrun.utils.get_in_artifact(artifact_obj, "tree"),
                    kind=mlrun.utils.get_in_artifact(artifact_obj, "kind"),
                    iter=mlrun.utils.get_in_artifact(artifact_obj, "iter"),
                )
            )
        return mlrun.common.schemas.TagObjects(
            kind="artifact", identifiers=artifact_identifiers
        )

    @abstractmethod
    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        pass

    @abstractmethod
    def store_project(
        self,
        name: str,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def create_project(
        self,
        project: mlrun.common.schemas.Project,
    ) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.name_only,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        pass

    @abstractmethod
    def get_project(self, name: str) -> mlrun.common.schemas.Project:
        pass

    @abstractmethod
    def list_artifact_tags(
        self,
        project=None,
        category: Union[str, mlrun.common.schemas.ArtifactCategories] = None,
    ):
        pass

    @abstractmethod
    def create_feature_set(
        self,
        feature_set: Union[dict, mlrun.common.schemas.FeatureSet],
        project="",
        versioned=True,
    ) -> dict:
        pass

    @abstractmethod
    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    # TODO: remove in 1.9.0
    @deprecated(
        version="1.9.0",
        reason="'list_features' will be removed in 1.9.0, use 'list_features_v2' instead",
        category=FutureWarning,
    )
    @abstractmethod
    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        pass

    @abstractmethod
    def list_features_v2(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutputV2:
        pass

    # TODO: remove in 1.9.0
    @deprecated(
        version="1.9.0",
        reason="'list_entities' will be removed in 1.9.0, use 'list_entities_v2' instead",
        category=FutureWarning,
    )
    @abstractmethod
    def list_entities(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutput:
        pass

    @abstractmethod
    def list_entities_v2(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutputV2:
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def delete_feature_set(self, name, project="", tag=None, uid=None):
        pass

    @abstractmethod
    def create_feature_vector(
        self,
        feature_vector: Union[dict, mlrun.common.schemas.FeatureVector],
        project="",
        versioned=True,
    ) -> dict:
        pass

    @abstractmethod
    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def delete_feature_vector(self, name, project="", tag=None, uid=None):
        pass

    @abstractmethod
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

    @abstractmethod
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
        pass

    @abstractmethod
    def create_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: dict = None,
    ):
        pass

    @abstractmethod
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

    @abstractmethod
    def list_project_secret_keys(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        token: str = None,
    ) -> mlrun.common.schemas.SecretKeysData:
        pass

    @abstractmethod
    def delete_project_secrets(
        self,
        project: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.kubernetes,
        secrets: list[str] = None,
    ):
        pass

    @abstractmethod
    def create_user_secrets(
        self,
        user: str,
        provider: Union[
            str, mlrun.common.schemas.SecretProviderName
        ] = mlrun.common.schemas.SecretProviderName.vault,
        secrets: dict = None,
    ):
        pass

    @abstractmethod
    def create_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        model_endpoint: Union[mlrun.model_monitoring.ModelEndpoint, dict],
    ):
        pass

    @abstractmethod
    def delete_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
    ):
        pass

    @abstractmethod
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

    @abstractmethod
    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        features: bool = False,
    ) -> mlrun.model_monitoring.ModelEndpoint:
        pass

    @abstractmethod
    def patch_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        attributes: dict,
    ):
        pass

    @abstractmethod
    def create_hub_source(
        self, source: Union[dict, mlrun.common.schemas.IndexedHubSource]
    ):
        pass

    @abstractmethod
    def store_hub_source(
        self,
        source_name: str,
        source: Union[dict, mlrun.common.schemas.IndexedHubSource],
    ):
        pass

    @abstractmethod
    def list_hub_sources(
        self,
        item_name: Optional[str] = None,
        tag: Optional[str] = None,
        version: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def get_hub_source(self, source_name: str):
        pass

    @abstractmethod
    def delete_hub_source(self, source_name: str):
        pass

    @abstractmethod
    def get_hub_catalog(
        self,
        source_name: str,
        version: str = None,
        tag: str = None,
        force_refresh: bool = False,
    ):
        pass

    @abstractmethod
    def get_hub_item(
        self,
        source_name: str,
        item_name: str,
        version: str = None,
        tag: str = "latest",
        force_refresh: bool = False,
    ):
        pass

    @abstractmethod
    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        pass

    @abstractmethod
    def store_api_gateway(
        self,
        api_gateway: Union[
            mlrun.common.schemas.APIGateway,
            "mlrun.runtimes.nuclio.api_gateway.APIGateway",
        ],
        project: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def list_api_gateways(self, project=None) -> mlrun.common.schemas.APIGatewaysOutput:
        pass

    @abstractmethod
    def get_api_gateway(self, name, project=None) -> mlrun.common.schemas.APIGateway:
        pass

    @abstractmethod
    def delete_api_gateway(self, name, project=None):
        pass

    @abstractmethod
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

    @abstractmethod
    def deploy_nuclio_function(
        self,
        func: "mlrun.runtimes.RemoteRuntime",
        builder_env: Optional[dict] = None,
    ):
        pass

    @abstractmethod
    def generate_event(
        self, name: str, event_data: Union[dict, mlrun.common.schemas.Event], project=""
    ):
        pass

    @abstractmethod
    def store_alert_config(
        self,
        alert_name: str,
        alert_data: Union[dict, mlrun.alerts.alert.AlertConfig],
        project="",
    ):
        pass

    @abstractmethod
    def get_alert_config(self, alert_name: str, project=""):
        pass

    @abstractmethod
    def list_alerts_configs(self, project=""):
        pass

    @abstractmethod
    def delete_alert_config(self, alert_name: str, project=""):
        pass

    @abstractmethod
    def reset_alert_config(self, alert_name: str, project=""):
        pass

    @abstractmethod
    def get_alert_template(self, template_name: str):
        pass

    @abstractmethod
    def list_alert_templates(self):
        pass

    @abstractmethod
    def get_builder_status(
        self,
        func: "mlrun.runtimes.BaseRuntime",
        offset: int = 0,
        logs: bool = True,
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        pass

    @abstractmethod
    def get_nuclio_deploy_status(
        self,
        func: "mlrun.runtimes.RemoteRuntime",
        last_log_timestamp: float = 0.0,
        verbose: bool = False,
    ):
        pass

    @abstractmethod
    def set_run_notifications(
        self,
        project: str,
        runs: list[mlrun.model.RunObject],
        notifications: list[mlrun.model.Notification],
    ):
        pass

    @abstractmethod
    def store_run_notifications(
        self,
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: str = None,
        mask_params: bool = True,
    ):
        pass

    @abstractmethod
    def get_log_size(self, uid, project=""):
        pass

    @abstractmethod
    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str,
        mask_params: bool = True,
    ):
        pass

    @abstractmethod
    def watch_log(self, uid, project="", watch=True, offset=0):
        pass

    @abstractmethod
    def get_datastore_profile(
        self, name: str, project: str
    ) -> Optional[mlrun.common.schemas.DatastoreProfile]:
        pass

    @abstractmethod
    def delete_datastore_profile(
        self, name: str, project: str
    ) -> mlrun.common.schemas.DatastoreProfile:
        pass

    @abstractmethod
    def list_datastore_profiles(
        self, project: str
    ) -> list[mlrun.common.schemas.DatastoreProfile]:
        pass

    @abstractmethod
    def store_datastore_profile(
        self, profile: mlrun.common.schemas.DatastoreProfile, project: str
    ):
        pass

    @abstractmethod
    def function_status(self, project, name, kind, selector):
        pass

    @abstractmethod
    def start_function(
        self, func_url: str = None, function: "mlrun.runtimes.BaseRuntime" = None
    ):
        pass

    @abstractmethod
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

    @abstractmethod
    def update_model_monitoring_controller(
        self,
        project: str,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
    ) -> None:
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def delete_model_monitoring_function(
        self, project: str, functions: list[str]
    ) -> bool:
        pass

    @abstractmethod
    def deploy_histogram_data_drift_app(
        self, project: str, image: str = "mlrun/mlrun"
    ) -> None:
        pass

    @abstractmethod
    def set_model_monitoring_credentials(
        self,
        project: str,
        credentials: dict[str, str],
        replace_creds: bool,
    ) -> None:
        pass
