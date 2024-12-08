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
from datetime import datetime

import pandas as pd
import sqlalchemy.orm

import mlrun.artifacts
import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.datastore
import mlrun.feature_store
import mlrun.model_monitoring
import mlrun.model_monitoring.helpers
from mlrun.model_monitoring.db._schedules import (
    ModelMonitoringSchedulesFile,
    delete_model_monitoring_schedules_folder,
)
from mlrun.model_monitoring.db._stats import (
    ModelMonitoringCurrentStatsFile,
    ModelMonitoringDriftMeasuresFile,
    delete_model_monitoring_stats_folder,
)
from mlrun.utils import logger

import framework.api.utils
import framework.utils.singletons.db
import services.api.crud.model_monitoring.deployment
import services.api.crud.model_monitoring.helpers
import services.api.crud.secrets


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

        logger.info(
            "Creating model endpoint",
            endpoint_id=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name=model_endpoint.spec.function_name,
        )

        # 1. store in db
        model_endpoint = framework.utils.singletons.db.get_db().store_model_endpoint(
            session=db_session,
            model_endpoint=model_endpoint,
            project=model_endpoint.metadata.project,
            name=model_endpoint.metadata.name,
            function_name=model_endpoint.spec.function_name,
        )

        # 2. according to the model uri get the model object
        # 3. get the feature stats from the model object
        model_obj = None
        if model_endpoint.spec.model_uri:
            model_endpoint, model_obj = cls._add_feature_stats(
                session=db_session, model_endpoint_object=model_endpoint
            )
        # # Verify and enrich the model endpoint obj with the updated model uri

        # 4. create a monitoring feature set & update the model endpoint object with
        # the feature_set_uri, features and labels
        # Get labels from model object if not found in model endpoint object
        features = []
        attributes = {}
        if model_obj:
            if not model_endpoint.spec.label_names and model_obj.spec.outputs:
                model_label_names = [
                    mlrun.feature_store.api.norm_column_name(f.name)
                    for f in model_obj.spec.outputs
                ]
                model_endpoint.spec.label_names = model_label_names
                attributes["label_names"] = model_label_names

            features = cls._get_features(
                model=model_obj,
                run_db=framework.api.utils.get_run_db_instance(db_session),
                project=model_endpoint.metadata.project,
            )
            model_endpoint.spec.feature_names = [feature.name for feature in features]
            attributes["feature_names"] = model_endpoint.spec.feature_names

        if (
            model_endpoint.status.monitoring_mode
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled
        ):
            monitoring_feature_set = cls.create_monitoring_feature_set(
                features=features,
                model_endpoint=model_endpoint,
                db_session=db_session,
            )
            # Link model endpoint object to feature set URI
            model_endpoint.spec.monitoring_feature_set_uri = monitoring_feature_set.uri
            attributes["monitoring_feature_set_uri"] = monitoring_feature_set.uri
            # Create model monitoring json files
            cls._create_model_monitoring_json_files(model_endpoint=model_endpoint)

        # 5. write the model endpoint to the db again
        framework.utils.singletons.db.get_db().update_model_endpoint(
            session=db_session,
            project=model_endpoint.metadata.project,
            name=model_endpoint.metadata.name,
            function_name=model_endpoint.spec.function_name,
            attributes=attributes,
            uid=model_endpoint.metadata.uid,
        )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Model endpoint created", endpoint_id=model_endpoint.metadata.uid)

        return model_endpoint

    @classmethod
    def _create_model_monitoring_json_files(
        cls, model_endpoint: mlrun.common.schemas.ModelEndpoint
    ):
        logger.info(
            "Creating model endpoint json files",
            model_endpoint_uid=model_endpoint.metadata.uid,
        )
        ModelMonitoringSchedulesFile.from_model_endpoint(
            model_endpoint=model_endpoint
        ).create()
        ModelMonitoringCurrentStatsFile.from_model_endpoint(
            model_endpoint=model_endpoint
        ).create()
        ModelMonitoringDriftMeasuresFile.from_model_endpoint(
            model_endpoint=model_endpoint
        ).create()

    def patch_model_endpoint(
        self,
        name: str,
        project: str,
        function_name: str,
        endpoint_id: str,
        attributes: dict,
        db_session: sqlalchemy.orm.Session,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """
        Update a model endpoint record with a given attributes.

        :param name: The name of the model endpoint.
        :param project: The name of the project.
        :param function_name: The name of the function.
        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the DB table. More details about the model
                           endpoint available attributes can be found under
                           :py:class:`~mlrun.common.schemas.ModelEndpoint`.
        :param db_session:             A session that manages the current dialog with the database.


        :return: A patched `ModelEndpoint` object without operative data.
        """

        model_endpoint = framework.utils.singletons.db.get_db().update_model_endpoint(
            session=db_session,
            project=project,
            name=name,
            function_name=function_name,
            attributes=attributes,
            uid=endpoint_id,
        )

        logger.info(
            "Model endpoint table updated",
            name=name,
            project=project,
            function_name=function_name,
            endpoint_id=model_endpoint.metadata.uid,
        )

        return model_endpoint

    @staticmethod
    def _get_features(
        model: mlrun.artifacts.ModelArtifact,
        project: str,
        run_db: mlrun.db.RunDBInterface,
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

        name = model_endpoint.metadata.name.replace(":", "-")

        feature_set = mlrun.feature_store.FeatureSet(
            f"monitoring-{serving_function_name}-{name}",
            entities=[
                mlrun.common.schemas.model_monitoring.FeatureSetFeatures.entity()
            ],
            timestamp_key=mlrun.common.schemas.model_monitoring.FeatureSetFeatures.time_stamp(),
            description=f"Monitoring feature set for endpoint: {model_endpoint.metadata.name}",
        )
        # Set the run db instance with the current db session
        feature_set._override_run_db(
            framework.api.utils.get_run_db_instance(db_session)
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
            services.api.crud.model_monitoring.helpers.get_monitoring_parquet_path(
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
            model_endpoint=model_endpoint.metadata.name,
            parquet_target=parquet_path,
        )

        return feature_set

    @staticmethod
    def delete_model_endpoint(
        name: str,
        project: str,
        function_name: str,
        endpoint_id: str,
        db_session: sqlalchemy.orm.Session,
    ) -> None:
        """
        Delete the record of a given model endpoint based on endpoint id.

        :param name:          The name of the model endpoint.
        :param project:       The name of the project.
        :param function_name: The name of the function.
        :param endpoint_id:   The unique id of the model endpoint.
        :param db_session:    A session that manages the current dialog with the database

        """
        if endpoint_id == "*":
            model_endpoint_list = (
                framework.utils.singletons.db.get_db().list_model_endpoints(
                    project=project,
                    name=name,
                    function_name=function_name,
                    latest_only=False,
                    session=db_session,
                )
            )
            uids = [
                model_endpoint.metadata.uid
                for model_endpoint in model_endpoint_list.endpoints
            ]
        else:
            uids = [endpoint_id]

        framework.utils.singletons.db.get_db().delete_model_endpoint(
            session=db_session,
            project=project,
            name=name,
            function_name=function_name,
            uid=endpoint_id,
        )
        # Delete the model endpoint files
        for uid in uids:
            ModelMonitoringCurrentStatsFile(project=project, endpoint_id=uid).delete()
            ModelMonitoringDriftMeasuresFile(project=project, endpoint_id=uid).delete()
            ModelMonitoringSchedulesFile(project=project, endpoint_id=uid).delete()

        logger.info(
            "Model endpoint were delete",
            project=project,
            name=name,
            function_name=function_name,
            amount=len(uids),
        )

    def get_model_endpoint(
        self,
        name: str,
        project: str,
        function_name: str,
        endpoint_id: str,
        tsdb_metrics: bool = True,
        feature_analysis: bool = False,
        db_session: sqlalchemy.orm.Session = None,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """Get a single model endpoint object.

        :param name                        The name of the model endpoint
        :param project:                    The name of the project
        :param function_name:              The name of the function
        :param endpoint_id:                The unique id of the model endpoint.
        :param tsdb_metrics:               When True, the time series metrics will be added to the output
                                           of the resulting.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.
        :param db_session:                 A session that manages the current dialog with the database.

        :return: A `ModelEndpoint` object.
        :raise: `MLRunNotFoundError` if the model endpoint is not found.
        """

        logger.info(
            "Getting model endpoint record from DB",
            name=name,
            project=project,
            function_name=function_name,
            endpoint_id=endpoint_id,
            tsdb_metrics=tsdb_metrics,
            feature_analysis=feature_analysis,
        )

        # Get the model endpoint record
        model_endpoint_object = (
            framework.utils.singletons.db.get_db().get_model_endpoint(
                session=db_session,
                project=project,
                name=name,
                function_name=function_name,
                uid=endpoint_id,
            )
        )

        # If time metrics were provided, retrieve the results from the time series DB
        if tsdb_metrics:
            logger.info("Adding real time metrics to the model endpoint")
            model_endpoint_object = self._add_basic_metrics(
                model_endpoint_objects=[model_endpoint_object],
                project=project,
            )[0]
        if feature_analysis:
            logger.info("Adding feature analysis to the model endpoint")
            model_endpoint_object = self._add_feature_analysis(
                model_endpoint_objects=[model_endpoint_object]
            )[0]
            if model_endpoint_object.spec.model_uri:
                model_endpoint_object, _ = self._add_feature_stats(
                    session=db_session, model_endpoint_object=model_endpoint_object
                )

        return model_endpoint_object

    def list_model_endpoints(
        self,
        project: str,
        db_session: sqlalchemy.orm.Session,
        name: typing.Optional[str] = None,
        model_name: typing.Optional[str] = None,
        function_name: typing.Optional[str] = None,
        labels: typing.Optional[list[str]] = None,
        start: typing.Optional[datetime] = None,
        end: typing.Optional[datetime] = None,
        top_level: typing.Optional[bool] = None,
        tsdb_metrics: typing.Optional[bool] = None,
        uids: typing.Optional[list[str]] = None,
        latest_only: typing.Optional[bool] = None,
    ) -> mlrun.common.schemas.ModelEndpointList:
        """
        List model endpoints based on the provided filters.
        :param project:             The name of the project.
        :param db_session:          A session that manages the current dialog with the database.
        :param name:                The name of the model endpoint.
        :param model_name:          The name of the model.
        :param function_name:       The name of the function.
        :param labels:              A list of labels to filter the model endpoints.
        :param start:               The start time of the model endpoint creation.
        :param end:                 The end time of the model endpoint creation.
        :param top_level:           When True, only top level model endpoints will be returned.
        :param tsdb_metrics:        When True, the time series metrics will be added to the output of the resulting
        :param uids:                A list of unique ids of the model endpoints.
        :param latest_only:         When True, only the latest model endpoint will be returned.
        :return:                    A list of `ModelEndpoint` objects.
        """

        logger.info(
            "Listing endpoints",
            name=name,
            project=project,
            model_name=model_name,
            function_name=function_name,
            labels=labels,
            start=start,
            end=end,
            top_level=top_level,
            tsdb_metrics=tsdb_metrics,
            uids=uids,
            latest_only=latest_only,
        )

        # Initialize an empty model endpoints list
        endpoint_list = framework.utils.singletons.db.get_db().list_model_endpoints(
            session=db_session,
            project=project,
            name=name,
            model_name=model_name,
            function_name=function_name,
            labels=labels,
            start=start,
            end=end,
            top_level=top_level,
            uids=uids,
            latest_only=latest_only,
        )

        if tsdb_metrics and endpoint_list.endpoints:
            endpoint_list.endpoints = self._add_basic_metrics(
                model_endpoint_objects=endpoint_list.endpoints,
                project=project,
            )

        return endpoint_list

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
        stream_path = services.api.crud.model_monitoring.get_stream_path(
            project=project_name,
        )
        if stream_path.startswith("v3io") and (
            not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api
        ):
            return
        elif stream_path.startswith("v3io") and not model_monitoring_access_key:
            # Generate V3IO Access Key
            try:
                model_monitoring_access_key = services.api.api.endpoints.nuclio.process_model_monitoring_secret(
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
            # Delete model monitoring TSDB resources
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=project_name,
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
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
        # Delete model monitoring stats folder.
        delete_model_monitoring_stats_folder(project=project_name)

        # Delete model monitoring schedules folder
        delete_model_monitoring_schedules_folder(project_name)

        logger.debug(
            "Successfully deleted model monitoring endpoints resources",
            project_name=project_name,
        )

    @staticmethod
    def get_model_endpoints_metrics(
        project: str,
        endpoint_id: str,
        type: str,
    ) -> list[mm_endpoints.ModelEndpointMonitoringMetric]:
        """
        Get the metrics for a given model endpoint.

        :param project:     The name of the project.
        :param endpoint_id: The unique id of the model endpoint.
        :param type:        metric or result.

        :return: A dictionary of metrics.
        """
        try:
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=project,
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
                    project=project
                ),
            )
        except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
            logger.debug(
                f"Failed to list model endpoint {type}s because tsdb connection is not defined."
                " Returning an empty list of metrics",
                error=mlrun.errors.err_to_str(e),
            )
            return []

        if type == "metric":
            df = tsdb_connector.get_metrics_metadata(endpoint_id=endpoint_id)
        elif type == "result":
            df = tsdb_connector.get_results_metadata(endpoint_id=endpoint_id)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Type must be either 'metric' or 'result'"
            )

        return tsdb_connector.df_to_metrics_list(df=df, type=type, project=project)

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
            services.api.crud.model_monitoring.deployment.MonitoringDeployment._delete_model_monitoring_stream_resources(
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
        len_of_feature_stats = len(model_endpoint.spec.feature_stats)

        if len_of_feature_stats != len_of_feature_names + len_of_label_names:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The length of model endpoint feature_stats is not equal to the "
                f"length of model endpoint feature names and labels "
                f"feature_stats({len_of_feature_stats}), "
                f"feature_names({len_of_feature_names}), "
                f"label_names({len_of_label_names}"
            )

    @staticmethod
    def _get_real_time_metrics(
        model_endpoint_object: mlrun.common.schemas.ModelEndpoint,
        metrics: typing.Optional[list[str]] = None,
        start: str = "now-1h",
        end: str = "now",
    ) -> dict[str, list[tuple[str, float]]]:
        """This method is working only with v3io tsdb, not in use for now.
           Add real time metrics from the time series DB to a provided `ModelEndpoint` object. The real time metrics
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
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
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

        return endpoint_metrics

    def _add_feature_analysis(
        self, model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint]
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add current stats and drift_measures to the model endpoint object.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will be filled with the relevant stats.

        :return: A list of `ModelEndpoint` objects.
        """
        for mep in model_endpoint_objects:
            mep.status.current_stats, mep.status.current_stats_timestamp = (
                ModelMonitoringCurrentStatsFile.from_model_endpoint(mep).read()
            )

            mep.status.drift_measures, mep.status.drift_measures_timestamp = (
                ModelMonitoringDriftMeasuresFile.from_model_endpoint(mep).read()
            )
        return model_endpoint_objects

    def _add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add basic metrics to the model endpoint object.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                        be filled with the relevant basic metrics.
        :param project:                The name of the project.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """

        def _add_metric(
            mep: mlrun.common.schemas.ModelEndpoint,
            df_dictionary: dict[str, pd.DataFrame],
        ):
            for metric in df_dictionary.keys():
                df = df_dictionary.get(metric, pd.DataFrame())
                if not df.empty:
                    line = df[df["endpoint_id"] == mep.metadata.uid]
                    if not line.empty and metric in line:
                        value = line[metric].item()
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        setattr(mep.status, metric, value)

            return mep

        try:
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=project,
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
                    project=project
                ),
            )
        except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
            logger.debug(
                "Failed to add basic metrics because tsdb connection is not defined."
                " Returning without adding basic metrics.",
                error=mlrun.errors.err_to_str(e),
            )
            return model_endpoint_objects

        uids = [mep.metadata.uid for mep in model_endpoint_objects]
        error_count_df = tsdb_connector.get_error_count(endpoint_ids=uids)
        last_request_df = tsdb_connector.get_last_request(endpoint_ids=uids)
        avg_latency_df = tsdb_connector.get_avg_latency(endpoint_ids=uids)
        drift_status_df = tsdb_connector.get_drift_status(endpoint_ids=uids)

        return list(
            map(
                lambda mep: _add_metric(
                    mep=mep,
                    df_dictionary={
                        "error_count": error_count_df,
                        "last_request": last_request_df,
                        "avg_latency": avg_latency_df,
                        "result_status": drift_status_df,
                    },
                ),
                model_endpoint_objects,
            )
        )

    @classmethod
    def _add_feature_stats(
        cls, session, model_endpoint_object: mlrun.common.schemas.ModelEndpoint
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, mlrun.artifacts.ModelArtifact]:
        """
        Add feature stats to the model endpoint object.

        :param session:                A session that manages the current dialog with the database.
        :param model_endpoint_object:  A `ModelEndpoint` object that will be filled with the relevant feature stats.

        :return: A list of `ModelEndpoint` objects.
        """

        run_db = framework.api.utils.get_run_db_instance(session)
        model_obj: mlrun.artifacts.ModelArtifact = (
            mlrun.datastore.store_resources.get_store_resource(
                model_endpoint_object.spec.model_uri, db=run_db
            )
        )
        feature_stats: dict = model_obj.spec.feature_stats or {}
        mlrun.common.model_monitoring.helpers.pad_features_hist(
            mlrun.common.model_monitoring.helpers.FeatureStats(feature_stats)
        )
        feature_stats.update(
            {
                mlrun.feature_store.api.norm_column_name(key): feature_stats.pop(key)
                for key in list(feature_stats.keys())
            }
        )

        model_endpoint_object.spec.feature_stats = feature_stats
        return model_endpoint_object, model_obj
