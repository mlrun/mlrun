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

import itertools
import typing

import sqlalchemy.orm

import mlrun.artifacts
import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.feature_store
import mlrun.model_monitoring
import mlrun.model_monitoring.helpers
import server.api.api.utils
import server.api.crud.model_monitoring.deployment
import server.api.crud.model_monitoring.helpers
import server.api.crud.secrets
import server.api.rundb.sqldb
from mlrun.utils import logger


class ModelEndpoints:
    """Provide different methods for handling model endpoints such as listing, writing and deleting"""

    @classmethod
    def create_model_endpoint(
        cls,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Creates model endpoint record in DB. The DB store target is defined either by a provided connection string
        or by the default store target that is defined in MLRun configuration.

        :param db_session:             A session that manages the current dialog with the database.
        :param model_endpoint:         Model endpoint object to update.

        :return: `ModelEndpoint` object.
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
            run_db = server.api.api.utils.get_run_db_instance(db_session)
            model_obj: mlrun.artifacts.ModelArtifact = (
                mlrun.datastore.store_resources.get_store_resource(
                    model_endpoint.spec.model_uri, db=run_db
                )
            )

            # Verify and enrich the model endpoint obj with the updated model uri
            mlrun.model_monitoring.helpers.enrich_model_endpoint_with_model_uri(
                model_endpoint=model_endpoint,
                model_obj=model_obj,
            )

            # Get stats from model object if not found in model endpoint object
            if not model_endpoint.status.feature_stats and hasattr(
                model_obj, "feature_stats"
            ):
                if model_obj.spec.feature_stats:
                    mlrun.common.model_monitoring.helpers.pad_features_hist(
                        mlrun.common.model_monitoring.helpers.FeatureStats(
                            model_obj.spec.feature_stats
                        )
                    )
                    model_endpoint.status.feature_stats = model_obj.spec.feature_stats
            # Get labels from model object if not found in model endpoint object
            if not model_endpoint.spec.label_names and model_obj.spec.outputs:
                model_label_names = [
                    mlrun.feature_store.api.norm_column_name(f.name)
                    for f in model_obj.spec.outputs
                ]
                model_endpoint.spec.label_names = model_label_names

            # Get algorithm from model object if not found in model endpoint object
            if not model_endpoint.spec.algorithm and model_obj.spec.algorithm:
                model_endpoint.spec.algorithm = model_obj.spec.algorithm

            features = cls._get_features(
                model=model_obj,
                run_db=run_db,
                project=model_endpoint.metadata.project,
            )
            model_endpoint.spec.feature_names = [feature.name for feature in features]
            # Create monitoring feature set if monitoring found in model endpoint object
            if (
                model_endpoint.spec.monitoring_mode
                == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled.value
            ):
                monitoring_feature_set = cls.create_monitoring_feature_set(
                    features=features,
                    model_endpoint=model_endpoint,
                    db_session=db_session,
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

            model_endpoint.status.feature_stats = cls._adjust_stats(
                model_endpoint=model_endpoint
            )

            logger.info(
                "Done preparing stats",
                feature_names=model_endpoint.spec.feature_names,
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Creating model endpoint", endpoint_id=model_endpoint.metadata.uid)
        # Write the new model endpoint
        model_endpoint_store = (
            server.api.crud.model_monitoring.helpers.get_store_object(
                project=model_endpoint.metadata.project
            )
        )
        model_endpoint_store.write_model_endpoint(endpoint=model_endpoint.flat_dict())

        logger.info("Model endpoint created", endpoint_id=model_endpoint.metadata.uid)

        return model_endpoint

    def patch_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        attributes: dict,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Update a model endpoint record with a given attributes.

        :param project: The name of the project.
        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the DB table. More details about the model
                           endpoint available attributes can be found under
                           :py:class:`~mlrun.common.schemas.ModelEndpoint`.

        :return: A patched `ModelEndpoint` object.
        """

        # Generate a model endpoint store object and apply the update process
        model_endpoint_store = (
            server.api.crud.model_monitoring.helpers.get_store_object(project=project)
        )
        model_endpoint_store.update_model_endpoint(
            endpoint_id=endpoint_id, attributes=attributes
        )

        logger.info("Model endpoint table updated", endpoint_id=endpoint_id)

        # Get the patched model endpoint record
        model_endpoint_record = model_endpoint_store.get_model_endpoint(
            endpoint_id=endpoint_id,
        )

        return self._convert_into_model_endpoint_object(endpoint=model_endpoint_record)

    @staticmethod
    def _get_features(
        model: mlrun.artifacts.ModelArtifact,
        project: str,
        run_db: server.api.rundb.sqldb.SQLRunDB,
    ) -> list[mlrun.feature_store.Feature]:
        """Get features to the feature set according to the model object"""
        features = []
        if model.spec.inputs:
            for feature in itertools.chain(model.spec.inputs, model.spec.outputs):
                name = mlrun.feature_store.api.norm_column_name(feature.name)
                features.append(
                    mlrun.feature_store.Feature(
                        name=name, value_type=feature.value_type
                    )
                )
        # Check if features can be found within the feature vector
        elif model.spec.feature_vector:
            _, name, _, tag, _ = mlrun.utils.helpers.parse_artifact_uri(
                model.spec.feature_vector
            )
            fv = run_db.get_feature_vector(name=name, project=project, tag=tag)
            for feature in fv.status.features:
                if feature["name"] != fv.status.label_column:
                    name = mlrun.feature_store.api.norm_column_name(feature["name"])
                    features.append(
                        mlrun.feature_store.Feature(
                            name=name, value_type=feature["value_type"]
                        )
                    )
        else:
            logger.warn(
                "Could not find any features in the model object and in the Feature Vector"
            )
        logger.debug("Listed features", features=features)
        return features

    @staticmethod
    def create_monitoring_feature_set(
        features: list[mlrun.feature_store.Feature],
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        db_session: sqlalchemy.orm.Session,
    ) -> mlrun.feature_store.FeatureSet:
        """
        Create monitoring feature set with the relevant parquet target.

        :param features:          The features list for the feature set.
        :param model_endpoint:    An object representing the model endpoint.
        :param db_session:        A session that manages the current dialog with the database.

        :return:                  Feature set object for the monitoring of the current model endpoint.
        """

        # append general features
        for feature in mlrun.common.schemas.model_monitoring.FeatureSetFeatures.list():
            features.append(mlrun.feature_store.Feature(name=feature))
        # Define a new feature set
        (
            _,
            serving_function_name,
            _,
            _,
        ) = mlrun.common.helpers.parse_versioned_object_uri(
            model_endpoint.spec.function_uri
        )

        model_name = model_endpoint.spec.model.replace(":", "-")

        feature_set = mlrun.feature_store.FeatureSet(
            f"monitoring-{serving_function_name}-{model_name}",
            entities=[
                mlrun.common.schemas.model_monitoring.FeatureSetFeatures.entity()
            ],
            timestamp_key=mlrun.common.schemas.model_monitoring.FeatureSetFeatures.time_stamp(),
            description=f"Monitoring feature set for endpoint: {model_endpoint.spec.model}",
        )
        # Set the run db instance with the current db session
        feature_set._override_run_db(
            server.api.api.utils.get_run_db_instance(db_session)
        )
        feature_set.spec.features = features
        feature_set.metadata.project = model_endpoint.metadata.project
        feature_set.metadata.labels = {
            mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID: model_endpoint.metadata.uid,
            mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_CLASS: model_endpoint.spec.model_class,
        }

        feature_set.metadata.tag = model_endpoint.metadata.uid + "_"

        # Define parquet target for this feature set
        parquet_path = (
            server.api.crud.model_monitoring.helpers.get_monitoring_parquet_path(
                db_session=db_session, project=model_endpoint.metadata.project
            )
            + f"/key={model_endpoint.metadata.uid}"
        )

        parquet_target = mlrun.datastore.targets.ParquetTarget(
            mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
            parquet_path,
        )
        driver = mlrun.datastore.targets.get_target_driver(parquet_target, feature_set)

        feature_set.set_targets(
            [mlrun.datastore.targets.ParquetTarget(path=parquet_path)],
            with_defaults=False,
        )
        driver.update_resource_status("created")

        # Save the new feature set
        feature_set.save()
        logger.info(
            "Monitoring feature set created",
            model_endpoint=model_endpoint.spec.model,
            parquet_target=parquet_path,
        )

        return feature_set

    @staticmethod
    def delete_model_endpoint(
        project: str,
        endpoint_id: str,
    ):
        """
        Delete the record of a given model endpoint based on endpoint id.

        :param project:     The name of the project.
        :param endpoint_id: The id of the endpoint.
        """
        model_endpoint_store = server.api.crud.ModelEndpoints()._get_store_object(
            project=project
        )
        if model_endpoint_store:
            model_endpoint_store.delete_model_endpoint(endpoint_id=endpoint_id)

            logger.info("Model endpoint table cleared", endpoint_id=endpoint_id)

    def get_model_endpoint(
        self,
        project: str,
        endpoint_id: str,
        metrics: list[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """Get a single model endpoint object. You can apply different time series metrics that will be added to the
           result.

        :param project:                    The name of the project
        :param endpoint_id:                The unique id of the model endpoint.
        :param metrics:                    A list of metrics to return for the model endpoint. There are pre-defined
                                           metrics for model endpoints such as predictions_per_second and
                                           latency_avg_5m but also custom metrics defined by the user. Please note that
                                           these metrics are stored in the time series DB and the results will be
                                           appeared under `model_endpoint.spec.metrics`.
        :param start:                      The start time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a  Unix timestamp in milliseconds, a relative time (`'now'`
                                           or `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days,
                                           and `'s'` = seconds), or 0 for the earliest time.
        :param end:                        The end time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a  Unix timestamp in milliseconds, a relative time (`'now'`
                                           or `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days,
                                           and `'s'` = seconds), or 0 for the earliest time.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.

        :return: A `ModelEndpoint` object.
        :raise: `MLRunNotFoundError` if the model endpoint is not found.
        """

        logger.info(
            "Getting model endpoint record from DB",
            endpoint_id=endpoint_id,
        )

        # Generate a model endpoint store object and get the model endpoint record as a dictionary
        model_endpoint_store = self._get_store_object(project=project)
        if model_endpoint_store:
            model_endpoint_record = model_endpoint_store.get_model_endpoint(
                endpoint_id=endpoint_id,
            )
        else:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Convert to `ModelEndpoint` object
        model_endpoint_object = self._convert_into_model_endpoint_object(
            endpoint=model_endpoint_record, feature_analysis=feature_analysis
        )

        # If time metrics were provided, retrieve the results from the time series DB
        if metrics:
            self._add_real_time_metrics(
                model_endpoint_object=model_endpoint_object,
                metrics=metrics,
                start=start,
                end=end,
            )

        return model_endpoint_object

    def list_model_endpoints(
        self,
        project: str,
        model: str = None,
        function: str = None,
        labels: list[str] = None,
        metrics: list[str] = None,
        start: str = "now-1h",
        end: str = "now",
        top_level: bool = False,
        uids: list[str] = None,
    ) -> mlrun.common.schemas.ModelEndpointList:
        """
        Returns a list of `ModelEndpoint` objects, wrapped in `ModelEndpointList` object. Each `ModelEndpoint`
        object represents the current state of a model endpoint. This functions supports filtering by the following
        parameters:
        1) model
        2) function
        3) labels
        4) top level
        5) uids
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters. By default, when the metrics parameter is None, no metrics are
        added to the output of this function.

        :param project:   The name of the project.
        :param model:     The name of the model to filter by.
        :param function:  The name of the function to filter by.
        :param labels:    A list of labels to filter by. Label filters work by either filtering a specific value of a
                          label (i.e. list("key=value")) or by looking for the existence of a given key (i.e. "key").
        :param metrics:   A list of metrics to return for each endpoint. There are pre-defined metrics for model
                          endpoints such as `predictions_per_second` and `latency_avg_5m` but also custom metrics
                          defined by the user. Please note that these metrics are stored in the time series DB and the
                          results will be appeared under model_endpoint.spec.metrics of each endpoint.
        :param start:     The start time of the metrics. Can be represented by a string containing an RFC 3339 time,
                          a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where `m`
                          = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
        :param end:       The end time of the metrics. Can be represented by a string containing an RFC 3339 time,
                          a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where `m`
                          = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
        :param top_level: If True, return only routers and endpoints that are NOT children of any router.
        :param uids:      List of model endpoint unique ids to include in the result.

        :return: An object of `ModelEndpointList` which is literally a list of model endpoints along with some metadata.
                 To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
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

        # Initialize an empty model endpoints list
        endpoint_list = mlrun.common.schemas.ModelEndpointList(endpoints=[])
        endpoint_store = self._get_store_object(project=project)
        if endpoint_store:
            endpoint_dictionary_list = endpoint_store.list_model_endpoints(
                function=function,
                model=model,
                labels=labels,
                top_level=top_level,
                uids=uids,
            )
        else:
            endpoint_dictionary_list = []

        for endpoint_dict in endpoint_dictionary_list:
            # Convert to `ModelEndpoint` object
            endpoint_obj = self._convert_into_model_endpoint_object(
                endpoint=endpoint_dict
            )

            # If time metrics were provided, retrieve the results from the time series DB
            if metrics:
                self._add_real_time_metrics(
                    model_endpoint_object=endpoint_obj,
                    metrics=metrics,
                    start=start,
                    end=end,
                )

            # Add the `ModelEndpoint` object into the model endpoints list
            endpoint_list.endpoints.append(endpoint_obj)

        return endpoint_list

    def verify_project_has_no_model_endpoints(self, project_name: str):
        """Verify that there no  model endpoint records in the DB by trying to list all of the project model endpoints.
        This method is usually being used during the process of deleting a project.

        :param project_name: project name.
        """

        if not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api:
            return
        endpoints = self.list_model_endpoints(project_name)
        if endpoints.endpoints:
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project_name} can not be deleted since related resources found: model endpoints"
            )

    def delete_model_endpoints_resources(
        self,
        project_name: str,
        db_session: sqlalchemy.orm.Session,
        model_monitoring_applications: typing.Optional[list[str]] = None,
        model_monitoring_access_key: typing.Optional[str] = None,
    ) -> None:
        """
        Delete all model endpoints resources, including the store data, time series data, and stream resources.

        :param project_name:                  The name of the project.
        :param db_session:                    A session that manages the current dialog with the database.
        :param model_monitoring_applications: A list of model monitoring applications that their resources should
                                              be deleted.
        :param model_monitoring_access_key:   The access key for the model monitoring resources. Relevant only for
                                              V3IO resources.
        """
        logger.debug(
            "Deleting model monitoring endpoints resources",
            project_name=project_name,
        )
        # We would ideally base on config.v3io_api but can't for backwards compatibility reasons,
        # we're using the igz version heuristic
        # TODO : adjust for ce scenario
        stream_path = server.api.crud.model_monitoring.get_stream_path(
            project=project_name,
        )
        if stream_path.startswith("v3io") and (
            not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api
        ):
            return
        elif stream_path.startswith("v3io") and not model_monitoring_access_key:
            # Generate V3IO Access Key
            try:
                model_monitoring_access_key = server.api.api.endpoints.nuclio.process_model_monitoring_secret(
                    db_session,
                    project_name,
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
                )

            except mlrun.errors.MLRunNotFoundError:
                logger.debug(
                    "Project does not exist in Iguazio, skipping deletion of model monitoring stream resources",
                    project_name=project_name,
                )
                return

        try:
            self.verify_project_has_no_model_endpoints(project_name=project_name)
        except mlrun.errors.MLRunPreconditionFailedError:
            # Delete model monitoring store resources
            endpoint_store = self._get_store_object(project=project_name)
            if endpoint_store:
                endpoint_store.delete_model_endpoints_resources()
            try:
                # Delete model monitoring TSDB resources
                tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                    project=project_name,
                    secret_provider=server.api.crud.secrets.get_project_secret_provider(
                        project=project_name
                    ),
                )
            except mlrun.errors.MLRunTSDBConnectionFailureError as e:
                logger.warning(
                    "Failed to delete TSDB resources, you may need to delete them manually",
                    project=project_name,
                    error=mlrun.errors.err_to_str(e),
                )
                tsdb_connector = None
            except mlrun.errors.MLRunInvalidMMStoreTypeError:
                # TODO: delete in 1.9.0 - for BC trying to delete from v3io store
                if not mlrun.mlconf.is_ce_mode():
                    tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                        project=project_name,
                        tsdb_connection_string=mlrun.common.schemas.model_monitoring.V3IO_MODEL_MONITORING_DB,
                    )
                else:
                    tsdb_connector = None
            if tsdb_connector:
                tsdb_connector.delete_tsdb_resources()
        self._delete_model_monitoring_stream_resources(
            project_name=project_name,
            model_monitoring_applications=model_monitoring_applications,
            model_monitoring_access_key=model_monitoring_access_key,
        )
        logger.debug(
            "Successfully deleted model monitoring endpoints resources",
            project_name=project_name,
        )

    @staticmethod
    def _delete_model_monitoring_stream_resources(
        project_name: str,
        model_monitoring_applications: typing.Optional[list[str]],
        model_monitoring_access_key: typing.Optional[str] = None,
    ) -> None:
        """
        Delete model monitoring stream resources.

        :param project_name:                  The name of the project.
        :param model_monitoring_applications: A list of model monitoring applications that their resources should
                                              be deleted.
        :param model_monitoring_access_key:   The access key for the model monitoring resources. Relevant only for
                                              V3IO resources.
        """
        logger.debug(
            "Deleting model monitoring stream resources",
            project_name=project_name,
        )

        model_monitoring_applications = model_monitoring_applications or []

        # Add the writer and monitoring stream to the application streams list
        model_monitoring_applications.append(
            mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.WRITER
        )
        model_monitoring_applications.append(
            mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.STREAM
        )

        try:
            server.api.crud.model_monitoring.deployment.MonitoringDeployment._delete_model_monitoring_stream_resources(
                project=project_name,
                function_names=model_monitoring_applications,
                access_key=model_monitoring_access_key,
            )
            logger.debug(
                "Successfully deleted model monitoring stream resources",
                project_name=project_name,
            )
        except mlrun.errors.MLRunStreamConnectionFailureError as e:
            logger.warning(
                "Failed to delete stream resources, you may need to delete them manually",
                project_name=project_name,
                function=model_monitoring_applications,
                error=mlrun.errors.err_to_str(e),
            )

    @staticmethod
    def _validate_length_features_and_labels(
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        """
        Validate that the length of feature_stats is equal to the length of `feature_names` and `label_names`

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
                f"feature_names({len_of_feature_names}), "
                f"label_names({len_of_label_names}"
            )

    @staticmethod
    def _adjust_stats(
        model_endpoint,
    ) -> mlrun.common.model_monitoring.helpers.FeatureStats:
        """
        Create a clean version of feature names for `feature_stats`.

        :param model_endpoint:    An object representing the model endpoint.
        :return: A Dictionary of feature stats with cleaned names
        """
        clean_feature_stats = {}
        for feature, stats in model_endpoint.status.feature_stats.items():
            clean_name = mlrun.feature_store.api.norm_column_name(feature)
            clean_feature_stats[clean_name] = stats
            # Exclude the label columns from the feature names
            if (
                model_endpoint.spec.label_names
                and clean_name in model_endpoint.spec.label_names
            ):
                continue
        return clean_feature_stats

    @staticmethod
    def _add_real_time_metrics(
        model_endpoint_object: mlrun.common.schemas.ModelEndpoint,
        metrics: list[str] = None,
        start: str = "now-1h",
        end: str = "now",
    ) -> mlrun.common.schemas.ModelEndpoint:
        """Add real time metrics from the time series DB to a provided `ModelEndpoint` object. The real time metrics
           will be stored under `ModelEndpoint.status.metrics.real_time`

        :param model_endpoint_object: `ModelEndpoint` object that will be filled with the relevant
                                       real time metrics.
        :param metrics:                A list of metrics to return for each endpoint. There are pre-defined metrics for
                                       model endpoints such as `predictions_per_second` and `latency_avg_5m` but also
                                       custom metrics defined by the user. Please note that these metrics are stored in
                                       the time series DB and the results will be appeared under
                                       model_endpoint.spec.metrics of each endpoint.
        :param start:                  The start time of the metrics. Can be represented by a string containing an RFC
                                       3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                       `'now-[0-9]+[mhd]'`, where `m`= minutes, `h` = hours, and `'d'` = days), or 0
                                       for the earliest time.
        :param end:                    The end time of the metrics. Can be represented by a string containing an RFC
                                       3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                       `'now-[0-9]+[mhd]'`, where `m`= minutes, `h` = hours, and `'d'` = days), or 0
                                       for the earliest time.

        """
        if model_endpoint_object.status.metrics is None:
            model_endpoint_object.status.metrics = {}

        try:
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=model_endpoint_object.metadata.project,
                secret_provider=server.api.crud.secrets.get_project_secret_provider(
                    project=model_endpoint_object.metadata.project
                ),
            )
        except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
            logger.debug(
                "Failed to add real time metrics because tsdb connection is not defined."
                " Returning without adding real time metrics.",
                error=mlrun.errors.err_to_str(e),
            )
            return model_endpoint_object

        endpoint_metrics = tsdb_connector.get_model_endpoint_real_time_metrics(
            endpoint_id=model_endpoint_object.metadata.uid,
            metrics=metrics,
            start=start,
            end=end,
        )

        if endpoint_metrics:
            model_endpoint_object.status.metrics[
                mlrun.common.schemas.model_monitoring.EventKeyMetrics.REAL_TIME
            ] = endpoint_metrics
        return model_endpoint_object

    @staticmethod
    def _convert_into_model_endpoint_object(
        endpoint: dict[str, typing.Any], feature_analysis: bool = False
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Create a `ModelEndpoint` object according to a provided model endpoint dictionary.

        :param endpoint:         Dictionary that represents a DB record of a model endpoint which need to be converted
                                 into a valid `ModelEndpoint` object.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A `~mlrun.common.schemas.ModelEndpoint` object.
        """

        # Convert into `ModelEndpoint` object
        endpoint_obj = mlrun.common.schemas.ModelEndpoint().from_flat_dict(endpoint)

        # If feature analysis was applied, add feature stats and current stats to the model endpoint result
        if feature_analysis and endpoint_obj.spec.feature_names:
            endpoint_features = (
                server.api.crud.model_monitoring.deployment.get_endpoint_features(
                    feature_names=endpoint_obj.spec.feature_names,
                    feature_stats=endpoint_obj.status.feature_stats,
                    current_stats=endpoint_obj.status.current_stats,
                )
            )
            if endpoint_features:
                endpoint_obj.status.features = endpoint_features
                # Add the latest drift measures results (calculated by the model monitoring batch)
                drift_measures = server.api.crud.model_monitoring.helpers.json_loads_if_not_none(
                    endpoint.get(
                        mlrun.common.schemas.model_monitoring.EventFieldType.DRIFT_MEASURES
                    )
                )
                endpoint_obj.status.drift_measures = drift_measures

        return endpoint_obj

    @staticmethod
    def _get_store_object(
        project: str,
    ) -> typing.Union[mlrun.model_monitoring.db.stores.base.store.StoreBase, None]:
        """
        Get the model endpoint store object.
        Firstly trying to use project secret and if there is no such secret
        it's trys to use the default/v3io store connection string.
        Note : Use this method only for deleting/reading model endpoints.
        """
        try:
            model_endpoint_store = (
                server.api.crud.model_monitoring.helpers.get_store_object(
                    project=project
                )
            )
        except mlrun.errors.MLRunInvalidMMStoreTypeError:
            # TODO: delete in 1.9.0 - for BC trying to create default/v3io store
            store_connection_string = (
                mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
                or mlrun.common.schemas.model_monitoring.V3IO_MODEL_MONITORING_DB
                if not mlrun.mlconf.is_ce_mode()
                else None
            )
            logger.debug(
                "Failed to create model endpoint store connector because store connection is not defined."
                " Trying use default/v3io."
            )
            model_endpoint_store = None
            if store_connection_string:
                model_endpoint_store = mlrun.model_monitoring.get_store_object(
                    project=project, store_connection_string=store_connection_string
                )
        return model_endpoint_store
