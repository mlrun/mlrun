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
import json
import os
import traceback
import typing

import nuclio.utils
import sqlalchemy.orm
import v3io.dataplane
import v3io_frames
import v3io_frames.errors

import mlrun.api.api.endpoints.functions
import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.schemas.model_endpoints
import mlrun.api.utils.singletons.k8s
import mlrun.artifacts
import mlrun.config
import mlrun.datastore.store_resources
import mlrun.errors
import mlrun.feature_store
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.model_monitoring.helpers
import mlrun.runtimes.function
import mlrun.utils.helpers
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.utils import logger


class ModelEndpoints:
    """Provide different methods for handling model endpoints such as listing, writing and deleting"""

    def create_or_patch(
        self,
        db_session: sqlalchemy.orm.Session,
        access_key: str,
        model_endpoint: mlrun.api.schemas.ModelEndpoint,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ):
        """
        :param db_session:             A session that manages the current dialog with the database
        :param access_key:             Access key with permission to write to KV table
        :param model_endpoint:         Model endpoint object to update
        :param auth_info:              The auth info of the request

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

        # If model artifact was supplied, grab model metadata from artifact
        if model_endpoint.spec.model_uri:
            logger.info(
                "Getting model object, inferring column names and collecting feature stats"
            )
            run_db = mlrun.api.api.utils.get_run_db_instance(db_session)
            model_obj: mlrun.artifacts.ModelArtifact = (
                mlrun.datastore.store_resources.get_store_resource(
                    model_endpoint.spec.model_uri, db=run_db
                )
            )

            # get stats from model object if not found in model endpoint object
            if not model_endpoint.status.feature_stats and hasattr(
                model_obj, "feature_stats"
            ):
                model_endpoint.status.feature_stats = model_obj.feature_stats

            # get labels from model object if not found in model endpoint object
            if not model_endpoint.spec.label_names and hasattr(model_obj, "outputs"):
                model_label_names = [
                    self._clean_feature_name(f.name) for f in model_obj.outputs
                ]
                model_endpoint.spec.label_names = model_label_names

            # get algorithm from model object if not found in model endpoint object
            if not model_endpoint.spec.algorithm and hasattr(model_obj, "algorithm"):
                model_endpoint.spec.algorithm = model_obj.algorithm

            # Create monitoring feature set if monitoring found in model endpoint object
            if (
                model_endpoint.spec.monitoring_mode
                == mlrun.api.schemas.ModelMonitoringMode.enabled.value
            ):
                monitoring_feature_set = self.create_monitoring_feature_set(
                    model_endpoint, model_obj, db_session, run_db
                )
                # Link model endpoint object to feature set URI
                model_endpoint.status.monitoring_feature_set_uri = (
                    monitoring_feature_set.uri
                )

        # If feature_stats was either populated by model_uri or by manual input, make sure to keep the names
        # of the features. If feature_names was supplied, replace the names set in feature_stats, otherwise - make
        # sure to keep a clean version of the names
        if model_endpoint.status.feature_stats:
            logger.info("Feature stats found, cleaning feature names")
            if model_endpoint.spec.feature_names:
                # Validate that the length of feature_stats is equal to the length of feature_names and label_names
                self._validate_length_features_and_labels(model_endpoint)

                # Clean feature names in both feature_stats and feature_names
            (
                model_endpoint.status.feature_stats,
                model_endpoint.spec.feature_names,
            ) = self._adjust_feature_names_and_stats(model_endpoint)

            logger.info(
                "Done preparing feature names and stats",
                feature_names=model_endpoint.spec.feature_names,
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Updating model endpoint", endpoint_id=model_endpoint.metadata.uid)

        self.write_endpoint_to_kv(
            access_key=access_key,
            endpoint=model_endpoint,
            update=True,
        )

        logger.info("Model endpoint updated", endpoint_id=model_endpoint.metadata.uid)

    def create_monitoring_feature_set(
        self,
        model_endpoint: mlrun.api.schemas.ModelEndpoint,
        model_obj: mlrun.artifacts.ModelArtifact,
        db_session: sqlalchemy.orm.Session,
        run_db: mlrun.db.sqldb.SQLDB,
    ):
        """
        Create monitoring feature set with the relevant parquet target.

        :param model_endpoint:    An object representing the model endpoint.
        :param model_obj:         An object representing the deployed model.
        :param db_session:        A session that manages the current dialog with the database.
        :param run_db:            A run db instance which will be used for retrieving the feature vector in case
                                  the features are not found in the model object.

        :return:                  Feature set object for the monitoring of the current model endpoint.
        """

        # Define a new feature set
        _, serving_function_name, _, _ = mlrun.utils.helpers.parse_versioned_object_uri(
            model_endpoint.spec.function_uri
        )

        model_name = model_endpoint.spec.model.replace(":", "-")

        feature_set = mlrun.feature_store.FeatureSet(
            f"monitoring-{serving_function_name}-{model_name}",
            entities=["endpoint_id"],
            timestamp_key="timestamp",
            description=f"Monitoring feature set for endpoint: {model_endpoint.spec.model}",
        )
        feature_set.metadata.project = model_endpoint.metadata.project

        feature_set.metadata.labels = {
            "endpoint_id": model_endpoint.metadata.uid,
            "model_class": model_endpoint.spec.model_class,
        }

        # Add features to the feature set according to the model object
        if model_obj.inputs.values():
            for feature in model_obj.inputs.values():
                feature_set.add_feature(
                    mlrun.feature_store.Feature(
                        name=feature.name, value_type=feature.value_type
                    )
                )
        # Check if features can be found within the feature vector
        elif model_obj.feature_vector:
            _, name, _, tag, _ = mlrun.utils.helpers.parse_artifact_uri(
                model_obj.feature_vector
            )
            fv = run_db.get_feature_vector(
                name=name, project=model_endpoint.metadata.project, tag=tag
            )
            for feature in fv.status.features:
                if feature["name"] != fv.status.label_column:
                    feature_set.add_feature(
                        mlrun.feature_store.Feature(
                            name=feature["name"], value_type=feature["value_type"]
                        )
                    )
        else:
            logger.warn(
                "Could not find any features in the model object and in the Feature Vector"
            )

        # Define parquet target for this feature set
        parquet_path = (
            f"v3io:///projects/{model_endpoint.metadata.project}"
            f"/model-endpoints/parquet/key={model_endpoint.metadata.uid}"
        )
        parquet_target = mlrun.datastore.targets.ParquetTarget("parquet", parquet_path)
        driver = mlrun.datastore.targets.get_target_driver(parquet_target, feature_set)
        driver.update_resource_status("created")
        feature_set.set_targets(
            [mlrun.datastore.targets.ParquetTarget(path=parquet_path)],
            with_defaults=False,
        )

        # Save the new feature set
        feature_set._override_run_db(db_session)
        feature_set.save()
        logger.info(
            "Monitoring feature set created",
            model_endpoint=model_endpoint.spec.model,
            parquet_target=parquet_path,
        )

        return feature_set

    @staticmethod
    def _validate_length_features_and_labels(model_endpoint):
        """
        Validate that the length of feature_stats is equal to the length of feature_names and label_names

        :param model_endpoint:    An object representing the model endpoint.
        """

        # Getting the length of label names, feature_names and feature_stats
        len_of_label_names = (
            0
            if not model_endpoint.spec.label_names
            else len(model_endpoint.spec.label_names)
        )
        len_of_feature_names = len(model_endpoint.spec.feature_names)
        len_of_feature_stats = len(model_endpoint.status.feature_stats)

        if len_of_feature_stats != len_of_feature_names + len_of_label_names:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The length of model endpoint feature_stats is not equal to the "
                f"length of model endpoint feature names and labels "
                f"feature_stats({len_of_feature_stats}), "
                f"feature_names({len_of_feature_names}),"
                f"label_names({len_of_label_names}"
            )

    def _adjust_feature_names_and_stats(
        self, model_endpoint
    ) -> typing.Tuple[typing.Dict, typing.List]:
        """
        Create a clean matching version of feature names for both feature_stats and feature_names. Please note that
        label names exist only in feature_stats and label_names.

        :param model_endpoint:    An object representing the model endpoint.
        :return: A tuple of:
             [0] = Dictionary of feature stats with cleaned names
             [1] = List of cleaned feature names
        """
        clean_feature_stats = {}
        clean_feature_names = []
        for i, (feature, stats) in enumerate(
            model_endpoint.status.feature_stats.items()
        ):
            clean_name = self._clean_feature_name(feature)
            clean_feature_stats[clean_name] = stats
            # Exclude the label columns from the feature names
            if (
                model_endpoint.spec.label_names
                and clean_name in model_endpoint.spec.label_names
            ):
                continue
            clean_feature_names.append(clean_name)
        return clean_feature_stats, clean_feature_names

    def delete_endpoint_record(
        self,
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        endpoint_id: str,
        access_key: str,
    ):
        """
        Deletes the KV record of a given model endpoint, project and endpoint_id are used for lookup

        :param auth_info: The auth info of the request
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param access_key: access key with permission to delete
        """
        logger.info("Clearing model endpoint table", endpoint_id=endpoint_id)
        client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api
        )

        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)

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
        model: typing.Optional[str] = None,
        function: typing.Optional[str] = None,
        labels: typing.Optional[typing.List[str]] = None,
        metrics: typing.Optional[typing.List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
        top_level: typing.Optional[bool] = False,
        uids: typing.Optional[typing.List[str]] = None,
    ) -> mlrun.api.schemas.model_endpoints.ModelEndpointList:
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

        :param auth_info: The auth info of the request
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

        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        if uids is None:
            client = mlrun.utils.v3io_clients.get_v3io_client(
                endpoint=mlrun.mlconf.v3io_api
            )

            path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
            )
            (
                _,
                container,
                path,
            ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)
            cursor = client.kv.new_cursor(
                container=container,
                table_path=path,
                access_key=auth_info.data_session,
                filter_expression=self.build_kv_cursor_filter_expression(
                    project,
                    function,
                    model,
                    labels,
                    top_level,
                ),
                attribute_names=["endpoint_id"],
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
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
        metrics: typing.Optional[typing.List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Returns a ModelEndpoint object with additional metrics and feature related data.

        :param auth_info: The auth info of the request
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
            "Getting model endpoint record from kv",
            endpoint_id=endpoint_id,
        )

        client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api
        )

        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)

        endpoint = client.kv.get(
            container=container,
            table_path=path,
            key=endpoint_id,
            access_key=access_key,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        )
        endpoint = endpoint.output.item

        if not endpoint:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

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

        endpoint = mlrun.api.schemas.ModelEndpoint(
            metadata=mlrun.api.schemas.ModelEndpointMetadata(
                project=endpoint.get("project"),
                labels=self._json_loads_if_not_none(labels),
                uid=endpoint_id,
            ),
            spec=mlrun.api.schemas.ModelEndpointSpec(
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
                monitoring_mode=endpoint.get("monitoring_mode") or None,
            ),
            status=mlrun.api.schemas.ModelEndpointStatus(
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
                monitoring_feature_set_uri=endpoint.get("monitoring_feature_set_uri")
                or None,
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
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.api.schemas.AuthInfo,
        tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
    ):
        """
        Invoking monitoring deploying functions.
        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """
        self.deploy_model_monitoring_stream_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auto_info=auth_info,
            tracking_policy=tracking_policy,
        )
        self.deploy_model_monitoring_batch_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
            tracking_policy=tracking_policy,
        )

    def write_endpoint_to_kv(
        self,
        access_key: str,
        endpoint: mlrun.api.schemas.ModelEndpoint,
        update: bool = True,
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

        client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api
        )
        function = client.kv.update if update else client.kv.put

        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=endpoint.metadata.project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)

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
                "monitoring_feature_set_uri": endpoint.status.monitoring_feature_set_uri
                or "",
                "monitoring_mode": endpoint.spec.monitoring_mode or "",
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
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
    ) -> typing.Dict[str, mlrun.api.schemas.Metric]:

        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names must be provided"
            )

        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=project, kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)

        client = mlrun.utils.v3io_clients.get_frames_client(
            token=access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=container,
        )

        metrics_mapping = {}

        try:
            data = client.read(
                backend="tsdb",
                table=path,
                columns=["endpoint_id", *metrics],
                filter=f"endpoint_id=='{endpoint_id}'",
                start=start,
                end=end,
            )

            data_dict = data.to_dict()
            for metric in metrics:
                metric_data = data_dict.get(metric)
                if metric_data is None:
                    continue

                values = [
                    (str(timestamp), value) for timestamp, value in metric_data.items()
                ]
                metrics_mapping[metric] = mlrun.api.schemas.Metric(
                    name=metric, values=values
                )
        except v3io_frames.errors.ReadError:
            logger.warn(f"failed to read tsdb for endpoint {endpoint_id}")
        return metrics_mapping

    def verify_project_has_no_model_endpoints(self, project_name: str):
        auth_info = mlrun.api.schemas.AuthInfo(
            data_session=os.getenv("V3IO_ACCESS_KEY")
        )

        if not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api:
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

        # We would ideally base on config.v3io_api but can't for backwards compatibility reasons,
        # we're using the igz version heuristic
        if not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api:
            return

        endpoints = self.list_endpoints(auth_info, project_name)
        for endpoint in endpoints.endpoints:
            self.delete_endpoint_record(
                auth_info,
                endpoint.metadata.project,
                endpoint.metadata.uid,
                access_key,
            )

        v3io_client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=access_key
        )

        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=project_name,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        tsdb_path = mlrun.utils.model_monitoring.parse_model_endpoint_project_prefix(
            path, project_name
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)

        frames = mlrun.utils.v3io_clients.get_frames_client(
            token=access_key,
            container=container,
            address=mlrun.mlconf.v3io_framesd,
        )
        try:
            all_records = v3io_client.kv.new_cursor(
                container=container,
                table_path=path,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
                access_key=access_key,
            ).all()

            all_records = [r["__name"] for r in all_records]

            # Cleanup KV
            for record in all_records:
                v3io_client.kv.delete(
                    container=container,
                    table_path=path,
                    key=record,
                    access_key=access_key,
                    raise_for_status=v3io.dataplane.RaiseForStatus.never,
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
                backend="tsdb",
                table=path,
                if_missing=v3io_frames.frames_pb2.IGNORE,
            )
        except v3io_frames.errors.CreateError:
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
        db_session: sqlalchemy.orm.Session,
        auto_info: mlrun.api.schemas.AuthInfo,
        tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
    ):
        """
        Deploying model monitoring stream real time nuclio function. The goal of this real time function is
        to monitor the log of the data stream. It is triggered when a new log entry is detected.
        It processes the new events into statistics that are then written to statistics databases.
        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """

        logger.info(
            "Checking if model monitoring stream is already deployed",
            project=project,
        )
        try:
            # validate that the model monitoring stream has not yet been deployed
            mlrun.runtimes.function.get_nuclio_deploy_status(
                name="model-monitoring-stream", project=project, tag=""
            )
            logger.info(
                "Detected model monitoring stream processing function already deployed",
                project=project,
            )
            return
        except nuclio.utils.DeployError:
            logger.info(
                "Deploying model monitoring stream processing function", project=project
            )

        fn = mlrun.model_monitoring.helpers.initial_model_monitoring_stream_processing_function(
            project, model_monitoring_access_key, db_session, tracking_policy
        )

        mlrun.api.api.endpoints.functions._build_function(
            db_session=db_session, auth_info=auto_info, function=fn
        )

    def deploy_model_monitoring_batch_processing(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.api.schemas.AuthInfo,
        tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
    ):
        """
        Deploying model monitoring batch job. The goal of this job is to identify drift in the data
        based on the latest batch of events. By default, this job is executed on the hour every hour.
        Note that if the monitoring batch job was already deployed then you will have to delete the
        old monitoring batch job before deploying a new one.
        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """

        logger.info(
            "Checking if model monitoring batch processing function is already deployed",
            project=project,
        )

        # Try to list functions that named model monitoring batch
        # to make sure that this job has not yet been deployed
        function_list = mlrun.api.utils.singletons.db.get_db().list_functions(
            session=db_session, name="model-monitoring-batch", project=project
        )

        if function_list:
            logger.info(
                "Detected model monitoring batch processing function already deployed",
                project=project,
            )
            return

        # Create a monitoring batch job function object
        fn = mlrun.model_monitoring.helpers.get_model_monitoring_batch_function(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
            tracking_policy=tracking_policy,
        )

        # Get the function uri
        function_uri = fn.save(versioned=True)
        function_uri = function_uri.replace("db://", "")

        task = mlrun.new_task(name="model-monitoring-batch", project=project)
        task.spec.function = function_uri

        # Apply batching interval params
        interval_list = [
            tracking_policy[
                model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
            ]["minute"],
            tracking_policy[
                model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
            ]["hour"],
            tracking_policy[
                model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
            ]["day"],
        ]
        minutes, hours, days = self._get_batching_interval_param(interval_list)
        batch_dict = {"minutes": minutes, "hours": hours, "days": days}

        task.spec.parameters[
            model_monitoring_constants.EventFieldType.BATCH_INTERVALS_DICT
        ] = batch_dict

        data = {
            "task": task.to_dict(),
            "schedule": self._convert_to_cron_string(
                tracking_policy[
                    model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
                ]
            ),
        }

        logger.info(
            "Deploying model monitoring batch processing function", project=project
        )

        # Add job schedule policy (every hour by default)
        mlrun.api.api.utils._submit_run(
            db_session=db_session, auth_info=auth_info, data=data
        )

    @staticmethod
    def get_endpoint_features(
        feature_names: typing.List[str],
        feature_stats: typing.Optional[dict],
        current_stats: typing.Optional[dict],
    ) -> typing.List[mlrun.api.schemas.Features]:
        safe_feature_stats = feature_stats or {}
        safe_current_stats = current_stats or {}

        features = []
        for name in feature_names:
            if feature_stats is not None and name not in feature_stats:
                logger.warn(f"Feature '{name}' missing from 'feature_stats'")
            if current_stats is not None and name not in current_stats:
                logger.warn(f"Feature '{name}' missing from 'current_stats'")
            f = mlrun.api.schemas.Features.new(
                name, safe_feature_stats.get(name), safe_current_stats.get(name)
            )
            features.append(f)
        return features

    @staticmethod
    def build_kv_cursor_filter_expression(
        project: str,
        function: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        labels: typing.Optional[typing.List[str]] = None,
        top_level: typing.Optional[bool] = False,
    ):
        if not project:
            raise mlrun.errors.MLRunInvalidArgumentError("project can't be empty")

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
                f"(endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)

    @staticmethod
    def _json_loads_if_not_none(field: typing.Any):
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
            raise mlrun.errors.MLRunBadRequestError("Data session is missing")
        return access_key

    @staticmethod
    def _get_batching_interval_param(intervals_list):
        """Converting each value in the intervals list into a float number. None
        Values will be converted into 0.0.

        example::
        Applying the function on a scheduling policy that is based on every hour exactly
        _get_batching_interval_param(intervals_list=[0, '*/1', None])
        The result will be: (0.0, 1.0, 0.0)

        """
        return tuple(
            map(
                lambda element: 0.0
                if isinstance(element, (float, int)) or element is None
                else float(f"0{element.partition('/')[-1]}"),
                intervals_list,
            )
        )

    @staticmethod
    def _convert_to_cron_string(cron_trigger):
        """Converting the batch interval dictionary into a ScheduleCronTrigger expression"""
        return "{} {} {} * *".format(
            cron_trigger["minute"], cron_trigger["hour"], cron_trigger["day"]
        ).replace("None", "*")
