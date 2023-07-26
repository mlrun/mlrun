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

import json
import typing
from datetime import datetime, timezone

import pandas as pd
import sqlalchemy as db

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.model_monitoring.helpers
from mlrun.common.db.sql_session import create_session, get_engine
from mlrun.utils import logger

from .model_endpoint_store import ModelEndpointStore
from .models import get_ModelEndpointsTable
from .models.base import Base


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
        sql_connection_string: str = None,
        secret_provider: typing.Callable = None,
    ):
        """
        Initialize SQL store target object.

        :param project:               The name of the project.
        :param sql_connection_string: Valid connection string or a path to SQL database with model endpoints table.
        :param secret_provider:       An optional secret provider to get the connection string secret.
        """

        super().__init__(project=project)

        self.sql_connection_string = (
            sql_connection_string
            or mlrun.model_monitoring.helpers.get_connection_string(
                secret_provider=secret_provider
            )
        )

        self.table_name = (
            mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_ENDPOINTS
        )

        self._engine = get_engine(dsn=self.sql_connection_string)
        self.ModelEndpointsTable = get_ModelEndpointsTable(
            connection_string=self.sql_connection_string
        )
        # Create table if not exist. The `metadata` contains the `ModelEndpointsTable`
        if not self._engine.has_table(self.table_name):
            Base.metadata.create_all(bind=self._engine)
        self.model_endpoints_table = self.ModelEndpointsTable.__table__

    def write_model_endpoint(self, endpoint: typing.Dict[str, typing.Any]):
        """
        Create a new endpoint record in the SQL table. This method also creates the model endpoints table within the
        SQL database if not exist.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """

        with self._engine.connect() as connection:
            # Adjust timestamps fields
            endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.FIRST_REQUEST
            ] = datetime.now(timezone.utc)
            endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.LAST_REQUEST
            ] = datetime.now(timezone.utc)

            # Convert the result into a pandas Dataframe and write it into the database
            endpoint_df = pd.DataFrame([endpoint])

            endpoint_df.to_sql(
                self.table_name, con=connection, index=False, if_exists="append"
            )

    def update_model_endpoint(
        self, endpoint_id: str, attributes: typing.Dict[str, typing.Any]
    ):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the SQL table.

        """

        # Update the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.sql_connection_string) as session:
            # Remove endpoint id (foreign key) from the update query
            attributes.pop(
                mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID, None
            )

            # Generate and commit the update session query
            session.query(self.ModelEndpointsTable).filter(
                self.ModelEndpointsTable.uid == endpoint_id
            ).update(attributes)
            session.commit()

    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the SQL record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """

        # Delete the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.sql_connection_string) as session:
            # Generate and commit the delete query
            session.query(self.ModelEndpointsTable).filter_by(uid=endpoint_id).delete()
            session.commit()

    def get_model_endpoint(
        self,
        endpoint_id: str,
    ) -> typing.Dict[str, typing.Any]:
        """
        Get a single model endpoint record.

        :param endpoint_id: The unique id of the model endpoint.

        :return: A model endpoint record as a dictionary.

        :raise MLRunNotFoundError: If the model endpoints table was not found or the model endpoint id was not found.
        """

        # Get the model endpoint record using sqlalchemy ORM
        with create_session(dsn=self.sql_connection_string) as session:
            # Generate the get query
            endpoint_record = (
                session.query(self.ModelEndpointsTable)
                .filter_by(uid=endpoint_id)
                .one_or_none()
            )

        if not endpoint_record:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Convert the database values and the table columns into a python dictionary
        return endpoint_record.to_dict()

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: typing.List[str] = None,
        top_level: bool = None,
        uids: typing.List = None,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Returns a list of model endpoint dictionaries, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available model endpoints for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key=value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param uids:             List of model endpoint unique ids to include in the result.

        :return: A list of model endpoint dictionaries.
        """

        # Generate an empty model endpoints that will be filled afterwards with model endpoint dictionaries
        endpoint_list = []

        # Get the model endpoints records using sqlalchemy ORM
        with create_session(dsn=self.sql_connection_string) as session:
            # Generate the list query
            query = session.query(self.ModelEndpointsTable).filter_by(
                project=self.project
            )

            # Apply filters
            if model:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.MODEL,
                    filtered_values=[model],
                )
            if function:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.FUNCTION,
                    filtered_values=[function],
                )
            if uids:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.UID,
                    filtered_values=uids,
                    combined=False,
                )
            if top_level:
                node_ep = str(
                    mlrun.common.schemas.model_monitoring.EndpointType.NODE_EP.value
                )
                router_ep = str(
                    mlrun.common.schemas.model_monitoring.EndpointType.ROUTER.value
                )
                endpoint_types = [node_ep, router_ep]
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=self.model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_TYPE,
                    filtered_values=endpoint_types,
                    combined=False,
                )
            # Convert the results from the DB into a ModelEndpoint object and append it to the model endpoints list
            for endpoint_record in query.all():
                endpoint_dict = endpoint_record.to_dict()

                # Filter labels
                if labels and not self._validate_labels(
                    endpoint_dict=endpoint_dict, labels=labels
                ):
                    continue

                endpoint_list.append(endpoint_dict)

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
            filter_query.append(model_endpoints_table.c[key_filter] == _filter)

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
            endpoint_dict.get(
                mlrun.common.schemas.model_monitoring.EventFieldType.LABELS
            )
        )

        for label in labels:
            # Case 1 - label is a key=value pair
            if "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                if lbl not in endpoint_labels or str(endpoint_labels[lbl]) != value:
                    return False
            # Case 2 - label is just a key
            else:
                if label not in endpoint_labels:
                    return False

        return True

    def delete_model_endpoints_resources(
        self, endpoints: typing.List[typing.Dict[str, typing.Any]]
    ):
        """
        Delete all model endpoints resources in both SQL and the time series DB.

        :param endpoints: A list of model endpoints flattened dictionaries.
        """

        for endpoint_dict in endpoints:
            # Delete model endpoint record from SQL table
            self.delete_model_endpoint(
                endpoint_dict[mlrun.common.schemas.model_monitoring.EventFieldType.UID],
            )

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> typing.Dict[str, typing.List[typing.Tuple[str, float]]]:
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
        # # TODO : Implement this method once Perometheus is supported
        logger.warning(
            "Real time metrics service using Prometheus will be implemented in 1.4.0"
        )

        return {}
