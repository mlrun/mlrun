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
import typing

import pandas as pd
import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base

import mlrun
import mlrun.api.schemas
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.api.db.sqldb.session import create_session, get_engine
from mlrun.api.utils.helpers import BaseModel
from mlrun.utils import logger

from .model_endpoint_store import ModelEndpointStore

Base = declarative_base()


class SQLModelEndpointStore(ModelEndpointStore):

    """
    Handles the DB operations when the DB target is from type SQL. For the SQL operations, we use SQLAlchemy, a Python
    SQL toolkit that handles the communication with the database.  When using SQL for storing the model endpoints
    record, the user needs to provide a valid connection string for the database.
    """

    _engine = None

    def __init__(
        self,
        project: str,
        connection_string: str = None,
    ):
        """
        Initialize SQL store target object.

        :param project: The name of the project.
        :param connection_string: Valid connection string or a path to SQL database with model endpoints table.
        """

        super().__init__(project=project)
        self.connection_string = (
            connection_string
            or mlrun.mlconf.model_endpoint_monitoring.connection_string
        )
        self.table_name = model_monitoring_constants.EventFieldType.MODEL_ENDPOINTS

        self._engine = get_engine(dsn=connection_string)

        # Create table if not exist. The `metadata` contains the `ModelEndpointsTable` defined later
        if not self._engine.has_table(self.table_name):
            Base.metadata.create_all(bind=self._engine)

        self.model_endpoints_table = ModelEndpointsTable.__table__

    def write_model_endpoint(self, endpoint: mlrun.api.schemas.ModelEndpoint):
        """
        Create a new endpoint record in the SQL table. This method also creates the model endpoints table within the
        SQL database if not exist.

        :param endpoint: `ModelEndpoint` object that will be written into the DB.
        """

        with self._engine.connect() as connection:

            # Retrieving the relevant attributes from the model endpoint object
            endpoint_dict = self.get_params(endpoint=endpoint)

            # Convert the result into a pandas Dataframe and write it into the database
            endpoint_df = pd.DataFrame([endpoint_dict])
            endpoint_df.to_sql(
                self.table_name, con=connection, index=False, if_exists="append"
            )

    def update_model_endpoint(self, endpoint_id: str, attributes: typing.Dict):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the SQL table.

        """

        # Update the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.connection_string) as session:

            # Generate and commit the update session query
            session.query(ModelEndpointsTable).filter(
                ModelEndpointsTable.endpoint_id == endpoint_id
            ).update(attributes)

            session.commit()

    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the SQL record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """

        # Delete the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.connection_string) as session:

            # Generate and commit the delete query
            session.query(ModelEndpointsTable).filter_by(
                endpoint_id=endpoint_id
            ).delete()
            session.commit()

    def get_model_endpoint(
        self,
        endpoint_id: str,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
        convert_to_endpoint_object: bool = True,
    ) -> typing.Union[mlrun.api.schemas.ModelEndpoint, dict]:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
        result.

        :param endpoint_id:                The unique id of the model endpoint.
        :param metrics:                    A list of real-time metrics to return for the model endpoint. There are
                                           pre-defined real-time metrics for model endpoints such as
                                           predictions_per_second and latency_avg_5m but also custom metrics defined
                                           by the user. Please note that these metrics are stored in the time series DB
                                           and the results will be appeared under model_endpoint.spec.metrics.
        :param start:                      The start time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                           `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or
                                           0 for the earliest time.
        :param end:                        The end time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                           `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                           or 0 for the earliest time.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.

        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a `ModelEndpoint` or not. True by default.

        :return: A `ModelEndpoint` object or a model endpoint dictionary if `convert_to_endpoint_object` is False.

        :raise MLRunNotFoundError: If the model endpoints table was not found or the model endpoint id was not found.
        """

        # Get the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.connection_string) as session:

            # Generate the get query
            endpoint_record = (
                session.query(ModelEndpointsTable)
                .filter_by(endpoint_id=endpoint_id)
                .one_or_none()
            )

        if not endpoint_record:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Convert the database values and the table columns into a python dictionary
        endpoint = endpoint_record.to_dict()

        if convert_to_endpoint_object:
            # Convert the model endpoint dictionary into a ModelEndpont object
            endpoint = self._convert_into_model_endpoint_object(
                endpoint=endpoint, feature_analysis=feature_analysis
            )

        # If time metrics were provided, retrieve the results from the time series DB
        if metrics:
            if endpoint.status.metrics is None:
                endpoint.status.metrics = {}

            endpoint_metrics = self.get_endpoint_real_time_metrics(
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                metrics=metrics,
            )
            if endpoint_metrics:
                endpoint.status.metrics[
                    model_monitoring_constants.EventKeyMetrics.REAL_TIME
                ] = endpoint_metrics

        return endpoint

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: typing.List[str] = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of `ModelEndpoint` objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available `ModelEndpoint` objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key=value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:         A list of real-time metrics to return for each model endpoint. There are pre-defined
                                real-time metrics for model endpoints such as predictions_per_second and latency_avg_5m
                                but also custom metrics defined by the user. Please note that these metrics are stored
                                in the time series DB and the results will be appeared under
                                `model_endpoint.spec.metrics`.
        :param start:           The start time of the metrics. Can be represented by a string containing an RFC 3339
                                time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for
                                 the earliest time.
        :param uids:             List of model endpoint unique ids to include in the result.

        :return: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """

        # Generate an empty ModelEndpointList that will be filled afterwards with ModelEndpoint objects
        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        # Get the model endpoints records using sqlalchemy ORM
        with create_session(dsn=self.connection_string) as session:

            # Generate the list query
            query = session.query(ModelEndpointsTable).filter_by(project=self.project)

            # Apply filters
            if model:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.MODEL,
                    filtered_values=[model],
                )
            if function:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.FUNCTION,
                    filtered_values=[function],
                )
            if uids:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.ENDPOINT_ID,
                    filtered_values=uids,
                    combined=False,
                )
            if top_level:
                node_ep = str(mlrun.model_monitoring.EndpointType.NODE_EP.value)
                router_ep = str(mlrun.model_monitoring.EndpointType.ROUTER.value)
                endpoint_types = [node_ep, router_ep]
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.ENDPOINT_TYPE,
                    filtered_values=endpoint_types,
                    combined=False,
                )

            # Convert the results from the DB into a ModelEndpoint object and append it to the ModelEndpointList
            for endpoint_record in query.all():
                endpoint_dict = endpoint_record.to_dict()

                # Filter labels
                if labels and not self._validate_labels(
                    endpoint_dict=endpoint_dict, labels=labels
                ):
                    continue

                endpoint_obj = self._convert_into_model_endpoint_object(endpoint_dict)

                # If time metrics were provided, retrieve the results from the time series DB
                if metrics:
                    if endpoint_obj.status.metrics is None:
                        endpoint_obj.status.metrics = {}
                    endpoint_metrics = self.get_endpoint_real_time_metrics(
                        endpoint_id=endpoint_obj.metadata.uid,
                        start=start,
                        end=end,
                        metrics=metrics,
                    )
                    if endpoint_metrics:

                        endpoint_obj.status.metrics[
                            model_monitoring_constants.EventKeyMetrics.REAL_TIME
                        ] = endpoint_metrics

                endpoint_list.endpoints.append(endpoint_obj)
        return endpoint_list

    @staticmethod
    def _filter_values(
        query: db.orm.query.Query,
        model_endpoints_table: db.Table,
        key_filter: str,
        filtered_values: typing.List,
        combined=True,
    ) -> db.orm.query.Query:
        """Filtering the SQL query object according to the provided filters.

        :param query:                 SQLAlchemy ORM query object. Includes the SELECT statements generated by the ORM
                                      for getting the model endpoint data from the SQL table.
        :param model_endpoints_table: SQLAlchemy table object that represents the model endpoints table.
        :param key_filter:            Key column to filter by.
        :param filtered_values:       List of values to filter the query the result.
        :param combined:              If true, then apply AND operator on the filtered values list. Otherwise, apply OR
                                      operator.

        return:                      SQLAlchemy ORM query object that represents the updated query with the provided
                                     filters.
        """

        if combined and len(filtered_values) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Can't apply combined policy with multiple values"
            )

        if not combined:
            return query.filter(
                model_endpoints_table.c[key_filter].in_(filtered_values)
            )

        # Generating a tuple with the relevant filters
        filter_query = []
        for _filter in filtered_values:
            filter_query.append(
                model_endpoints_table.c[key_filter] == _filter,
            )

        # Apply AND operator on the SQL query object with the filters tuple
        return query.filter(db.and_(*filter_query))

    @staticmethod
    def _validate_labels(
        endpoint_dict: dict,
        labels: typing.List,
    ) -> bool:
        """Validate that the model endpoint dictionary has the provided labels. There are 2 possible cases:
        1 - Labels were provided as a list of key-values pairs (e.g. ['label_1=value_1', 'label_2=value_2']): Validate
            that each pair exist in the endpoint dictionary.
        2 - Labels were provided as a list of key labels (e.g. ['label_1', 'label_2']): Validate that each key exist in
            the endpoint labels dictionary.

        :param endpoint_dict: Dictionary of the model endpoint records.
        :param labels:        List of dictionary of required labels.

        :return: True if the labels exist in the endpoint labels dictionary, otherwise False.
        """

        # Convert endpoint labels into dictionary
        endpoint_labels = json.loads(
            endpoint_dict.get(model_monitoring_constants.EventFieldType.LABELS)
        )

        for label in labels:
            # Case 1 - labels are a list of key-value pairs
            if "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                if lbl not in endpoint_labels or str(endpoint_labels[lbl]) != value:
                    return False
            # Case 2 - labels are a list of keys
            else:
                if label not in endpoint_labels:
                    return False

        return True

    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        """
        Delete all model endpoints resources in both SQL and the time series DB. In addition, delete the model
        endpoints table from SQL if it's empty at the end of the process.

        :param endpoints: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """

        # Delete model endpoint record from SQL table
        for endpoint in endpoints.endpoints:
            self.delete_model_endpoint(
                endpoint.metadata.uid,
            )

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> typing.Dict[str, typing.List]:
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user.

        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param access_key:       V3IO access key that will be used for generating Frames client object. If not
                                 provided, the access key will be retrieved from the environment variables.

        :return: A dictionary of metrics in which the key is a metric name and the value is a list of tuples that
                 includes timestamps and the values.
        """
        # TODO : Implement this method once Perometheus is supported
        logger.warning(
            "Real time metrics service using Prometheus will be implemented in 1.4.0"
        )
        return {}


class ModelEndpointsTable(Base, BaseModel):
    __tablename__ = model_monitoring_constants.EventFieldType.MODEL_ENDPOINTS

    endpoint_id = db.Column(
        model_monitoring_constants.EventFieldType.ENDPOINT_ID,
        db.String(40),
        primary_key=True,
    )
    state = db.Column(model_monitoring_constants.EventFieldType.STATE, db.String(10))
    project = db.Column(
        model_monitoring_constants.EventFieldType.PROJECT, db.String(40)
    )
    function_uri = db.Column(
        model_monitoring_constants.EventFieldType.FUNCTION_URI,
        db.String(255),
    )
    model = db.Column(model_monitoring_constants.EventFieldType.MODEL, db.String(255))
    model_class = db.Column(
        model_monitoring_constants.EventFieldType.MODEL_CLASS,
        db.String(255),
    )
    labels = db.Column(model_monitoring_constants.EventFieldType.LABELS, db.Text)
    model_uri = db.Column(
        model_monitoring_constants.EventFieldType.MODEL_URI, db.String(255)
    )
    stream_path = db.Column(
        model_monitoring_constants.EventFieldType.STREAM_PATH, db.Text
    )
    algorithm = db.Column(
        model_monitoring_constants.EventFieldType.ALGORITHM,
        db.String(255),
    )
    active = db.Column(model_monitoring_constants.EventFieldType.ACTIVE, db.Boolean)
    monitoring_mode = db.Column(
        model_monitoring_constants.EventFieldType.MONITORING_MODE,
        db.String(10),
    )
    feature_stats = db.Column(
        model_monitoring_constants.EventFieldType.FEATURE_STATS, db.Text
    )
    current_stats = db.Column(
        model_monitoring_constants.EventFieldType.CURRENT_STATS, db.Text
    )
    feature_names = db.Column(
        model_monitoring_constants.EventFieldType.FEATURE_NAMES, db.Text
    )
    children = db.Column(model_monitoring_constants.EventFieldType.CHILDREN, db.Text)
    label_names = db.Column(
        model_monitoring_constants.EventFieldType.LABEL_NAMES, db.Text
    )
    timestamp = db.Column(
        model_monitoring_constants.EventFieldType.TIMESTAMP,
        db.DateTime,
    )
    endpoint_type = db.Column(
        model_monitoring_constants.EventFieldType.ENDPOINT_TYPE,
        db.String(10),
    )
    children_uids = db.Column(
        model_monitoring_constants.EventFieldType.CHILDREN_UIDS, db.Text
    )
    drift_measures = db.Column(
        model_monitoring_constants.EventFieldType.DRIFT_MEASURES, db.Text
    )
    drift_status = db.Column(
        model_monitoring_constants.EventFieldType.DRIFT_STATUS,
        db.String(40),
    )
    monitor_configuration = db.Column(
        model_monitoring_constants.EventFieldType.MONITOR_CONFIGURATION,
        db.Text,
    )
    monitoring_feature_set_uri = db.Column(
        model_monitoring_constants.EventFieldType.FEATURE_SET_URI,
        db.String(255),
    )
    first_request = db.Column(
        model_monitoring_constants.EventFieldType.FIRST_REQUEST,
        db.String(40),
    )
    last_request = db.Column(
        model_monitoring_constants.EventFieldType.LAST_REQUEST,
        db.String(40),
    )
    error_count = db.Column(
        model_monitoring_constants.EventFieldType.ERROR_COUNT, db.Integer
    )
    metrics = db.Column(model_monitoring_constants.EventFieldType.METRICS, db.Text)
