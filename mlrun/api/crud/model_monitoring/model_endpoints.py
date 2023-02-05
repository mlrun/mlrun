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


import os
import typing

import sqlalchemy.orm

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

from .model_endpoint_store import get_model_endpoint_target


class ModelEndpoints:
    """Provide different methods for handling model endpoints such as listing, writing and deleting"""

    def create_or_patch(
        self,
        db_session: sqlalchemy.orm.Session,
        access_key: str,
        model_endpoint: mlrun.api.schemas.ModelEndpoint,
        auth_info: mlrun.api.schemas.AuthInfo = mlrun.api.schemas.AuthInfo(),
    ) -> mlrun.api.schemas.ModelEndpoint:
        # TODO: deprecated, remove in 1.5.0.
        """
        Either create or updates the record of a given ModelEndpoint object.
        Leaving here for backwards compatibility, remove in 1.5.0.

        :param db_session:             A session that manages the current dialog with the database
        :param access_key:             Access key with permission to write to KV table
        :param model_endpoint:         Model endpoint object to update
        :param auth_info:              The auth info of the request

        :return: Model endpoint object.
        """

        return self.create_model_endpoint(
            db_session=db_session, model_endpoint=model_endpoint
        )

    def create_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.api.schemas.ModelEndpoint,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Creates model endpoint record in DB. The DB target type is defined under
        mlrun.config.model_endpoint_monitoring.store_type (KV by default).

        :param db_session:             A session that manages the current dialog with the database.
        :param model_endpoint:         Model endpoint object to update.

        :return: Model endpoint object.
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

            # Get stats from model object if not found in model endpoint object
            if not model_endpoint.status.feature_stats and hasattr(
                model_obj, "feature_stats"
            ):
                model_endpoint.status.feature_stats = model_obj.feature_stats

            # Get labels from model object if not found in model endpoint object
            if not model_endpoint.spec.label_names and hasattr(model_obj, "outputs"):
                model_label_names = [
                    self._clean_feature_name(f.name) for f in model_obj.outputs
                ]
                model_endpoint.spec.label_names = model_label_names

            # Get algorithm from model object if not found in model endpoint object
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
            ) = self._adjust_feature_names_and_stats(model_endpoint=model_endpoint)

            logger.info(
                "Done preparing feature names and stats",
                feature_names=model_endpoint.spec.feature_names,
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Creating model endpoint", endpoint_id=model_endpoint.metadata.uid)

        # Write the new model endpoint
        model_endpoint_target = get_model_endpoint_target(
            project=model_endpoint.metadata.project,
        )
        model_endpoint_target.write_model_endpoint(endpoint=model_endpoint)

        logger.info("Model endpoint created", endpoint_id=model_endpoint.metadata.uid)

        return model_endpoint

    @staticmethod
    def create_monitoring_feature_set(
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

    @staticmethod
    def patch_model_endpoint(
        project: str,
        endpoint_id: str,
        attributes: dict,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Update a model endpoint record with a given attributes.

        :param project: The name of the project.
        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table. More details about the model
                           endpoint available attributes can be found under
                           :py:class:`~mlrun.api.schemas.ModelEndpoint`.

        :return: A patched ModelEndpoint object.
        """

        model_endpoint_target = get_model_endpoint_target(
            project=project,
        )
        model_endpoint_target.update_model_endpoint(
            endpoint_id=endpoint_id, attributes=attributes
        )

        return model_endpoint_target.get_model_endpoint(
            endpoint_id=endpoint_id, start="now-1h", end="now"
        )

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
        model_endpoint_target = get_model_endpoint_target(
            project=project,
        )
        model_endpoint_target.delete_model_endpoint(endpoint_id=endpoint_id)

    @staticmethod
    def get_model_endpoint(
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        endpoint_id: str,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """Get a single model endpoint object. You can apply different time series metrics that will be added to the
           result.

        :param auth_info: The auth info of the request
        :param project: The name of the project
        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined metrics for
                                 model endpoints such as predictions_per_second and latency_avg_5m but also custom
                                 metrics defined by the user. Please note that these metrics are stored in the time
                                 series DB and the results will be appeared under model_endpoint.spec.metrics.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` =
                                 days), or 0 for the earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` =
                                 days), or 0 for the earliest time.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A ModelEndpoint object.
        """

        model_endpoint_target = get_model_endpoint_target(
            project=project, access_key=auth_info.data_session
        )
        return model_endpoint_target.get_model_endpoint(
            endpoint_id=endpoint_id,
            metrics=metrics,
            start=start,
            end=end,
            feature_analysis=feature_analysis,
        )

    @staticmethod
    def list_model_endpoints(
        auth_info: mlrun.api.schemas.AuthInfo,
        project: str,
        model: str = None,
        function: str = None,
        labels: typing.List[str] = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        top_level: bool = False,
        uids: typing.List[str] = None,
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

        :param auth_info: The auth info of the request.
        :param project:   The name of the project.
        :param model:     The name of the model to filter by.
        :param function:  The name of the function to filter by.
        :param labels:    A list of labels to filter by. Label filters work by either filtering a specific value of a
                          label (i.e. list("key==value")) or by looking for the existence of a given key (i.e. "key").
        :param metrics:   A list of metrics to return for each endpoint. There are pre-defined metrics for model
                          endpoints such as predictions_per_second and latency_avg_5m but also custom metrics defined
                          by the user. Please note that these metrics are stored in the time series DB and the results
                          will be appeared under model_endpoint.spec.metrics of each endpoint.
        :param start:     The start time of the metrics. Can be represented by a string containing an RFC 3339 time,
                          a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where `m`
                          = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
        :param end:       The end time of the metrics. Can be represented by a string containing an RFC 3339 time,
                          a Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where `m`
                          = minutes, `h` = hours, and `'d'` = days), or 0 for the earliest time.
        :param top_level: If True will return only routers and endpoint that are NOT children of any router.
        :param uids:      Will return ModelEndpointList of endpoints with uid in uids.

        :return: An object of ModelEndpointList which is literally a list of model endpoints along with some metadata.
                 To get a standard list of model endpoints use ModelEndpointList.endpoints.
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

        endpoint_target = get_model_endpoint_target(
            access_key=auth_info.data_session, project=project
        )

        # Initialize an empty model endpoints list
        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        # If list of model endpoint ids was not provided, retrieve it from the DB
        if uids is None:
            uids = endpoint_target.list_model_endpoints(
                function=function, model=model, labels=labels, top_level=top_level
            )

        # Add each relevant model endpoint to the model endpoints list
        for endpoint_id in uids:
            endpoint = endpoint_target.get_model_endpoint(
                metrics=metrics,
                endpoint_id=endpoint_id,
                start=start,
                end=end,
            )
            endpoint_list.endpoints.append(endpoint)

        return endpoint_list

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
            auth_info=auth_info,
            tracking_policy=tracking_policy,
        )
        self.deploy_model_monitoring_batch_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
            tracking_policy=tracking_policy,
        )

    def verify_project_has_no_model_endpoints(self, project_name: str):
        auth_info = mlrun.api.schemas.AuthInfo(
            data_session=os.getenv("V3IO_ACCESS_KEY")
        )

        if not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api:
            return

        endpoints = self.list_model_endpoints(auth_info, project_name)
        if endpoints.endpoints:
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project_name} can not be deleted since related resources found: model endpoints"
            )

    def delete_model_endpoints_resources(self, project_name: str):
        """
        Delete all model endpoints resources.

        :param project_name: The name of the project.
        """
        auth_info = mlrun.api.schemas.AuthInfo(
            data_session=os.getenv("V3IO_ACCESS_KEY")
        )

        # We would ideally base on config.v3io_api but can't for backwards compatibility reasons,
        # we're using the igz version heuristic
        if not mlrun.mlconf.igz_version or not mlrun.mlconf.v3io_api:
            return

        endpoints = self.list_model_endpoints(auth_info, project_name)

        endpoint_target = get_model_endpoint_target(
            access_key=auth_info.data_session, project=project_name
        )
        endpoint_target.delete_model_endpoints_resources(endpoints)

    @staticmethod
    def deploy_model_monitoring_stream_processing(
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.api.schemas.AuthInfo,
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
                name="model-monitoring-stream",
                project=project,
                tag="",
                auth_info=auth_info,
            )
            logger.info(
                "Detected model monitoring stream processing function already deployed",
                project=project,
            )
            return
        except mlrun.errors.MLRunNotFoundError:
            logger.info(
                "Deploying model monitoring stream processing function", project=project
            )

        fn = mlrun.model_monitoring.helpers.initial_model_monitoring_stream_processing_function(
            project, model_monitoring_access_key, db_session, tracking_policy
        )

        mlrun.api.api.endpoints.functions._build_function(
            db_session=db_session, auth_info=auth_info, function=fn
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
        mlrun.api.api.utils.submit_run_sync(
            db_session=db_session, auth_info=auth_info, data=data
        )

    @staticmethod
    def _clean_feature_name(feature_name):
        return feature_name.replace(" ", "_").replace("(", "").replace(")", "")

    @staticmethod
    def get_access_key(auth_info: mlrun.api.schemas.AuthInfo):
        """
        Getting access key from the current data session. This method is usually used to verify that the session
        is valid and contains an access key.

        param auth_info: The auth info of the request.

        :return: Access key as a string.
        """
        access_key = auth_info.data_session
        if not access_key:
            raise mlrun.errors.MLRunBadRequestError("Data session is missing")
        return access_key

    @staticmethod
    def _get_batching_interval_param(intervals_list: typing.List):
        """Converting each value in the intervals list into a float number. None
        Values will be converted into 0.0.

        param intervals_list: A list of values based on the ScheduleCronTrigger expression. Note that at the moment
                              it supports minutes, hours, and days. e.g. [0, '*/1', None] represents on the hour
                              every hour.

        :return: A tuple of:
                 [0] = minutes interval as a float
                 [1] = hours interval as a float
                 [2] = days interval as a float
        """
        return tuple(
            [
                0.0
                if isinstance(interval, (float, int)) or interval is None
                else float(f"0{interval.partition('/')[-1]}")
                for interval in intervals_list
            ]
        )

    @staticmethod
    def _convert_to_cron_string(cron_trigger):
        """Converting the batch interval dictionary into a ScheduleCronTrigger expression"""
        return "{} {} {} * *".format(
            cron_trigger["minute"], cron_trigger["hour"], cron_trigger["day"]
        ).replace("None", "*")
