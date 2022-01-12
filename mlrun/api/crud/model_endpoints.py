import json
import os
import traceback
from typing import Any, Dict, List, Optional

from nuclio.utils import DeployError
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import frames_pb2
from v3io_frames.errors import CreateError

import mlrun.api.api.utils
import mlrun.api.utils.singletons.k8s
import mlrun.datastore.store_resources
from mlrun.api.api.endpoints.functions import _build_function
from mlrun.api.api.utils import _submit_run, get_run_db_instance
from mlrun.api.crud.secrets import Secrets
from mlrun.api.schemas import (
    Features,
    Metric,
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from mlrun.api.schemas.model_endpoints import ModelEndpointList
from mlrun.api.utils.singletons.db import get_db
from mlrun.artifacts import ModelArtifact
from mlrun.config import config
from mlrun.errors import (
    MLRunBadRequestError,
    MLRunInvalidArgumentError,
    MLRunNotFoundError,
)
from mlrun.model_monitoring.helpers import (
    get_model_monitoring_stream_processing_function,
)
from mlrun.runtimes import KubejobRuntime
from mlrun.runtimes.function import get_nuclio_deploy_status
from mlrun.utils.helpers import logger
from mlrun.utils.model_monitoring import (
    EndpointType,
    parse_model_endpoint_project_prefix,
    parse_model_endpoint_store_prefix,
)
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client


class ModelEndpoints:
    def create_or_patch(
        self,
        db_session: Session,
        access_key: str,
        model_endpoint: ModelEndpoint,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        """
        Creates or patch a KV record with the given model_endpoint record
        """

        if model_endpoint.spec.model_uri or model_endpoint.status.feature_stats:
            logger.info(
                "Getting feature metadata",
                project=model_endpoint.metadata.project,
                model=model_endpoint.spec.model,
                function=model_endpoint.spec.function_uri,
                model_uri=model_endpoint.spec.model_uri,
            )

        # If model artifact was supplied, grab model meta data from artifact
        if model_endpoint.spec.model_uri:
            logger.info(
                "Getting model object, inferring column names and collecting feature stats"
            )
            run_db = mlrun.api.api.utils.get_run_db_instance(db_session)
            model_obj: ModelArtifact = (
                mlrun.datastore.store_resources.get_store_resource(
                    model_endpoint.spec.model_uri, db=run_db
                )
            )

            if not model_endpoint.status.feature_stats and hasattr(
                model_obj, "feature_stats"
            ):
                model_endpoint.status.feature_stats = model_obj.feature_stats

            if not model_endpoint.spec.label_names and hasattr(model_obj, "outputs"):
                model_label_names = [
                    self._clean_feature_name(f.name) for f in model_obj.outputs
                ]
                model_endpoint.spec.label_names = model_label_names

            if not model_endpoint.spec.algorithm and hasattr(model_obj, "algorithm"):
                model_endpoint.spec.algorithm = model_obj.algorithm

        # If feature_stats was either populated by model_uri or by manual input, make sure to keep the names
        # of the features. If feature_names was supplied, replace the names set in feature_stats, otherwise - make
        # sure to keep a clean version of the names
        if model_endpoint.status.feature_stats:
            logger.info("Feature stats found, cleaning feature names")
            if model_endpoint.spec.feature_names:
                if len(model_endpoint.status.feature_stats) != len(
                    model_endpoint.spec.feature_names
                ):
                    raise MLRunInvalidArgumentError(
                        f"feature_stats and feature_names have a different number of names, while expected to match"
                        f"feature_stats({len(model_endpoint.status.feature_stats)}), "
                        f"feature_names({len(model_endpoint.spec.feature_names)})"
                    )
            clean_feature_stats = {}
            clean_feature_names = []
            for i, (feature, stats) in enumerate(
                model_endpoint.status.feature_stats.items()
            ):
                if model_endpoint.spec.feature_names:
                    clean_name = self._clean_feature_name(
                        model_endpoint.spec.feature_names[i]
                    )
                else:
                    clean_name = self._clean_feature_name(feature)
                clean_feature_stats[clean_name] = stats
                clean_feature_names.append(clean_name)
            model_endpoint.status.feature_stats = clean_feature_stats
            model_endpoint.spec.feature_names = clean_feature_names

            logger.info(
                "Done preparing feature names and stats",
                feature_names=model_endpoint.spec.feature_names,
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Updating model endpoint", endpoint_id=model_endpoint.metadata.uid)

        self.write_endpoint_to_kv(
            access_key=access_key, endpoint=model_endpoint, update=True,
        )

        logger.info("Model endpoint updated", endpoint_id=model_endpoint.metadata.uid)

        return model_endpoint

    def delete_endpoint_record(
        self,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        endpoint_id: str,
        access_key: str,
    ):
        """
        Deletes the KV record of a given model endpoint, project and endpoint_id are used for lookup

        :param auth_info: The required auth information for doing the deletion
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param access_key: access key with permission to delete
        """
        logger.info("Clearing model endpoint table", endpoint_id=endpoint_id)
        client = get_v3io_client(endpoint=config.v3io_api)

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=project, kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        client.kv.delete(
            container=container,
            table_path=path,
            key=endpoint_id,
            access_key=access_key,
        )

        logger.info("Model endpoint table cleared", endpoint_id=endpoint_id)

    def list_endpoints(
        self,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        labels: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
        top_level: Optional[bool] = False,
        uids: Optional[List[str]] = None,
    ) -> ModelEndpointList:
        """
        Returns a list of ModelEndpointState objects. Each object represents the current state of a model endpoint.
        This functions supports filtering by the following parameters:
        1) model
        2) function
        3) labels
        4) top level
        5) uids
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters. By default, when the metrics parameter is None, no metrics are
        added to the output of this function.

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param model: The name of the model to filter by
        :param function: The name of the function to filter by
        :param labels: A list of labels to filter by. Label filters work by either filtering a specific value of a label
        (i.e. list("key==value")) or by looking for the existence of a given key (i.e. "key")
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        :param top_level: if True will return only routers and endpoint that are NOT children of any router
        :param uids: will return ModelEndpointList of endpoints with uid in uids
        """

        logger.info(
            "Listing endpoints",
            project=project,
            model=model,
            function=function,
            labels=labels,
            metrics=metrics,
            start=start,
            end=end,
            top_level=top_level,
            uids=uids,
        )

        endpoint_list = ModelEndpointList(endpoints=[])

        if uids is None:
            client = get_v3io_client(endpoint=config.v3io_api)

            path = config.model_endpoint_monitoring.store_prefixes.default.format(
                project=project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
            )
            _, container, path = parse_model_endpoint_store_prefix(path)
            cursor = client.kv.new_cursor(
                container=container,
                table_path=path,
                access_key=auth_info.data_session,
                filter_expression=self.build_kv_cursor_filter_expression(
                    project, function, model, labels, top_level,
                ),
                attribute_names=["endpoint_id"],
                raise_for_status=RaiseForStatus.never,
            )
            try:
                items = cursor.all()
            except Exception:
                return endpoint_list

            uids = [item["endpoint_id"] for item in items]

        for endpoint_id in uids:
            endpoint = self.get_endpoint(
                auth_info=auth_info,
                project=project,
                endpoint_id=endpoint_id,
                metrics=metrics,
                start=start,
                end=end,
            )
            endpoint_list.endpoints.append(endpoint)
        return endpoint_list

    def get_endpoint(
        self,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        endpoint_id: str,
        metrics: Optional[List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
    ) -> ModelEndpoint:
        """
        Returns a ModelEndpoint object with additional metrics and feature related data.

        :param auth_info: The required auth information for doing the deletion
        :param project: The name of the project
        :param endpoint_id: The id of the model endpoint
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
        the output of the resulting object
        """
        access_key = self.get_access_key(auth_info)
        logger.info(
            "Getting model endpoint record from kv", endpoint_id=endpoint_id,
        )

        client = get_v3io_client(endpoint=config.v3io_api)

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=project, kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        endpoint = client.kv.get(
            container=container,
            table_path=path,
            key=endpoint_id,
            access_key=access_key,
            raise_for_status=RaiseForStatus.never,
        )
        endpoint = endpoint.output.item

        if not endpoint:
            raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        labels = endpoint.get("labels")

        feature_names = endpoint.get("feature_names")
        feature_names = self._json_loads_if_not_none(feature_names)

        label_names = endpoint.get("label_names")
        label_names = self._json_loads_if_not_none(label_names)

        feature_stats = endpoint.get("feature_stats")
        feature_stats = self._json_loads_if_not_none(feature_stats)

        current_stats = endpoint.get("current_stats")
        current_stats = self._json_loads_if_not_none(current_stats)

        drift_measures = endpoint.get("drift_measures")
        drift_measures = self._json_loads_if_not_none(drift_measures)

        children = endpoint.get("children")
        children = self._json_loads_if_not_none(children)

        monitor_configuration = endpoint.get("monitor_configuration")
        monitor_configuration = self._json_loads_if_not_none(monitor_configuration)

        endpoint_type = endpoint.get("endpoint_type")
        endpoint_type = self._json_loads_if_not_none(endpoint_type)

        children_uids = endpoint.get("children_uids")
        children_uids = self._json_loads_if_not_none(children_uids)

        endpoint = ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=endpoint.get("project"),
                labels=self._json_loads_if_not_none(labels),
                uid=endpoint_id,
            ),
            spec=ModelEndpointSpec(
                function_uri=endpoint.get("function_uri"),
                model=endpoint.get("model"),
                model_class=endpoint.get("model_class") or None,
                model_uri=endpoint.get("model_uri") or None,
                feature_names=feature_names or None,
                label_names=label_names or None,
                stream_path=endpoint.get("stream_path") or None,
                algorithm=endpoint.get("algorithm") or None,
                monitor_configuration=monitor_configuration or None,
                active=endpoint.get("active") or None,
            ),
            status=ModelEndpointStatus(
                state=endpoint.get("state") or None,
                feature_stats=feature_stats or None,
                current_stats=current_stats or None,
                children=children or None,
                first_request=endpoint.get("first_request") or None,
                last_request=endpoint.get("last_request") or None,
                accuracy=endpoint.get("accuracy") or None,
                error_count=endpoint.get("error_count") or None,
                drift_status=endpoint.get("drift_status") or None,
                endpoint_type=endpoint_type or None,
                children_uids=children_uids or None,
            ),
        )

        if feature_analysis and feature_names:
            endpoint_features = self.get_endpoint_features(
                feature_names=feature_names,
                feature_stats=feature_stats,
                current_stats=current_stats,
            )
            if endpoint_features:
                endpoint.status.features = endpoint_features
                endpoint.status.drift_measures = drift_measures

        if metrics:
            endpoint_metrics = self.get_endpoint_metrics(
                access_key=access_key,
                project=project,
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=metrics,
            )
            if endpoint_metrics:
                endpoint.status.metrics = endpoint_metrics

        return endpoint

    def deploy_monitoring_functions(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
        self.deploy_model_monitoring_stream_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auto_info=auth_info,
        )
        self.deploy_model_monitoring_batch_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
        )

    def write_endpoint_to_kv(
        self, access_key: str, endpoint: ModelEndpoint, update: bool = True
    ):
        """
        Writes endpoint data to KV, a prerequisite for initializing the monitoring process

        :param access_key: V3IO access key for managing user permissions
        :param endpoint: ModelEndpoint object
        :param update: When True, use client.kv.update, otherwise use client.kv.put
        """

        labels = endpoint.metadata.labels or {}
        searchable_labels = {f"_{k}": v for k, v in labels.items()} if labels else {}

        feature_names = endpoint.spec.feature_names or []
        label_names = endpoint.spec.label_names or []
        feature_stats = endpoint.status.feature_stats or {}
        current_stats = endpoint.status.current_stats or {}
        children = endpoint.status.children or []
        monitor_configuration = endpoint.spec.monitor_configuration or {}
        endpoint_type = endpoint.status.endpoint_type or None
        children_uids = endpoint.status.children_uids or []

        client = get_v3io_client(endpoint=config.v3io_api)
        function = client.kv.update if update else client.kv.put

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=endpoint.metadata.project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        function(
            container=container,
            table_path=path,
            key=endpoint.metadata.uid,
            access_key=access_key,
            attributes={
                "endpoint_id": endpoint.metadata.uid,
                "project": endpoint.metadata.project,
                "function_uri": endpoint.spec.function_uri,
                "model": endpoint.spec.model,
                "model_class": endpoint.spec.model_class or "",
                "labels": json.dumps(labels),
                "model_uri": endpoint.spec.model_uri or "",
                "stream_path": endpoint.spec.stream_path or "",
                "active": endpoint.spec.active or "",
                "state": endpoint.status.state or "",
                "feature_stats": json.dumps(feature_stats),
                "current_stats": json.dumps(current_stats),
                "feature_names": json.dumps(feature_names),
                "children": json.dumps(children),
                "label_names": json.dumps(label_names),
                "monitor_configuration": json.dumps(monitor_configuration),
                "endpoint_type": json.dumps(endpoint_type),
                "children_uids": json.dumps(children_uids),
                **searchable_labels,
            },
        )

        return endpoint

    def get_endpoint_metrics(
        self,
        access_key: str,
        project: str,
        endpoint_id: str,
        metrics: List[str],
        start: str = "now-1h",
        end: str = "now",
    ) -> Dict[str, Metric]:

        if not metrics:
            raise MLRunInvalidArgumentError("Metric names must be provided")

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=project, kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS
        )
        _, container, path = parse_model_endpoint_store_prefix(path)

        client = get_frames_client(
            token=access_key, address=config.v3io_framesd, container=container,
        )

        data = client.read(
            backend="tsdb",
            table=path,
            columns=["endpoint_id", *metrics],
            filter=f"endpoint_id=='{endpoint_id}'",
            start=start,
            end=end,
        )

        data_dict = data.to_dict()
        metrics_mapping = {}
        for metric in metrics:
            metric_data = data_dict.get(metric)
            if metric_data is None:
                continue

            values = [
                (str(timestamp), value) for timestamp, value in metric_data.items()
            ]
            metrics_mapping[metric] = Metric(name=metric, values=values)
        return metrics_mapping

    def verify_project_has_no_model_endpoints(self, project_name: str):
        auth_info = mlrun.api.schemas.AuthInfo(
            data_session=os.getenv("V3IO_ACCESS_KEY")
        )

        if not config.igz_version or not config.v3io_api:
            return

        endpoints = self.list_endpoints(auth_info, project_name)
        if endpoints.endpoints:
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project_name} can not be deleted since related resources found: model endpoints"
            )

    def delete_model_endpoints_resources(self, project_name: str):
        auth_info = mlrun.api.schemas.AuthInfo(
            data_session=os.getenv("V3IO_ACCESS_KEY")
        )
        access_key = auth_info.data_session

        # we would ideally base on config.v3io_api but can't for backwards compatibility reasons,
        # we're using the igz version heuristic
        if not config.igz_version or not config.v3io_api:
            return

        endpoints = self.list_endpoints(auth_info, project_name)
        for endpoint in endpoints.endpoints:
            self.delete_endpoint_record(
                auth_info, endpoint.metadata.project, endpoint.metadata.uid, access_key,
            )

        v3io = get_v3io_client(endpoint=config.v3io_api, access_key=access_key)

        path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=project_name,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        tsdb_path = parse_model_endpoint_project_prefix(path, project_name)
        _, container, path = parse_model_endpoint_store_prefix(path)

        frames = get_frames_client(
            token=access_key, container=container, address=config.v3io_framesd,
        )
        try:
            all_records = v3io.kv.new_cursor(
                container=container,
                table_path=path,
                raise_for_status=RaiseForStatus.never,
                access_key=access_key,
            ).all()

            all_records = [r["__name"] for r in all_records]

            # Cleanup KV
            for record in all_records:
                v3io.kv.delete(
                    container=container,
                    table_path=path,
                    key=record,
                    access_key=access_key,
                    raise_for_status=RaiseForStatus.never,
                )
        except RuntimeError as exc:
            # KV might raise an exception even it was set not raise one.  exception is raised if path is empty or
            # not exist, therefore ignoring failures until they'll fix the bug.
            # TODO: remove try except after bug is fixed
            logger.debug(
                "Failed cleaning model endpoints KV. Ignoring",
                exc=str(exc),
                traceback=traceback.format_exc(),
            )
            pass

        # Cleanup TSDB
        try:
            frames.delete(
                backend="tsdb", table=path, if_missing=frames_pb2.IGNORE,
            )
        except CreateError:
            # frames might raise an exception if schema file does not exist.
            pass

        # final cleanup of tsdb path
        tsdb_path.replace("://u", ":///u")
        store, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    @staticmethod
    def deploy_model_monitoring_stream_processing(
        project: str,
        model_monitoring_access_key: str,
        db_session,
        auto_info: mlrun.api.schemas.AuthInfo,
    ):
        logger.info(
            f"Checking deployment status for model monitoring stream processing function [{project}]"
        )
        try:
            get_nuclio_deploy_status(
                name="model-monitoring-stream", project=project, tag=""
            )
            logger.info(
                f"Detected model monitoring stream processing function [{project}] already deployed"
            )
            return
        except DeployError:
            logger.info(
                f"Deploying model monitoring stream processing function [{project}]"
            )

        fn = get_model_monitoring_stream_processing_function(
            project, model_monitoring_access_key, db_session
        )

        _build_function(db_session=db_session, auth_info=auto_info, function=fn)

    @staticmethod
    def deploy_model_monitoring_batch_processing(
        project: str,
        model_monitoring_access_key: str,
        db_session,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
        logger.info(
            f"Checking deployment status for model monitoring batch processing function [{project}]"
        )
        function_list = get_db().list_functions(
            session=db_session, name="model-monitoring-batch", project=project
        )

        if function_list:
            logger.info(
                f"Detected model monitoring batch processing function [{project}] already deployed"
            )
            return

        logger.info(f"Deploying model monitoring batch processing function [{project}]")

        fn: KubejobRuntime = mlrun.import_function(
            f"hub://model_monitoring_batch:{config.model_endpoint_monitoring.batch_processing_function_branch}"
        )

        fn.set_db_connection(get_run_db_instance(db_session))

        fn.metadata.project = project

        fn.apply(mlrun.mount_v3io())

        fn.set_env_from_secret(
            "MODEL_MONITORING_ACCESS_KEY",
            mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
            Secrets().generate_model_monitoring_secret_key(
                "MODEL_MONITORING_ACCESS_KEY"
            ),
        )

        # Needs to be a member of the project and have access to project data path
        fn.metadata.credentials.access_key = model_monitoring_access_key

        function_uri = fn.save(versioned=True)
        function_uri = function_uri.replace("db://", "")

        task = mlrun.new_task(name="model-monitoring-batch", project=project)
        task.spec.function = function_uri

        data = {
            "task": task.to_dict(),
            "schedule": "0 */1 * * *",
        }

        _submit_run(db_session=db_session, auth_info=auth_info, data=data)

    @staticmethod
    def get_endpoint_features(
        feature_names: List[str],
        feature_stats: Optional[dict],
        current_stats: Optional[dict],
    ) -> List[Features]:
        safe_feature_stats = feature_stats or {}
        safe_current_stats = current_stats or {}

        features = []
        for name in feature_names:
            if feature_stats is not None and name not in feature_stats:
                logger.warn(f"Feature '{name}' missing from 'feature_stats'")
            if current_stats is not None and name not in current_stats:
                logger.warn(f"Feature '{name}' missing from 'current_stats'")
            f = Features.new(
                name, safe_feature_stats.get(name), safe_current_stats.get(name)
            )
            features.append(f)
        return features

    @staticmethod
    def build_kv_cursor_filter_expression(
        project: str,
        function: Optional[str] = None,
        model: Optional[str] = None,
        labels: Optional[List[str]] = None,
        top_level: Optional[bool] = False,
    ):
        if not project:
            raise MLRunInvalidArgumentError("project can't be empty")

        filter_expression = [f"project=='{project}'"]

        if function:
            filter_expression.append(f"function=='{function}'")
        if model:
            filter_expression.append(f"model=='{model}'")
        if labels:
            for label in labels:

                if not label.startswith("_"):
                    label = f"_{label}"

                if "=" in label:
                    lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                    filter_expression.append(f"{lbl}=='{value}'")
                else:
                    filter_expression.append(f"exists({label})")
        if top_level:
            filter_expression.append(
                f"(endpoint_type=='{str(EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)

    @staticmethod
    def _json_loads_if_not_none(field: Any):
        if field is None:
            return None
        return json.loads(field)

    @staticmethod
    def _clean_feature_name(feature_name):
        return feature_name.replace(" ", "_").replace("(", "").replace(")", "")

    @staticmethod
    def get_access_key(auth_info: mlrun.api.schemas.AuthInfo):
        access_key = auth_info.data_session
        if not access_key:
            raise MLRunBadRequestError("Data session is missing")
        return access_key
