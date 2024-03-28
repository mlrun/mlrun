# Copyright 2024 Iguazio
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
import json
import typing
import uuid

import pandas as pd
import sqlalchemy

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.model_monitoring.db
import mlrun.model_monitoring.db.stores.sqldb.models
import mlrun.model_monitoring.helpers
from mlrun.common.db.sql_session import create_session, get_engine
from mlrun.utils import logger


class SQLStoreBase(mlrun.model_monitoring.db.StoreBase):
    """
    Handles the DB operations when the DB target is from type SQL. For the SQL operations, we use SQLAlchemy, a Python
    SQL toolkit that handles the communication with the database.  When using SQL for storing the model monitoring
    data, the user needs to provide a valid connection string for the database.
    """

    _engine = None
    _tables = {}

    def __init__(
        self,
        project: str,
        secret_provider: typing.Callable = None,
    ):
        """
        Initialize SQL store target object.

        :param project:               The name of the project.
        :param secret_provider:       An optional secret provider to get the connection string secret.
        """

        super().__init__(project=project)

        self._sql_connection_string = (
            mlrun.model_monitoring.helpers.get_connection_string(
                secret_provider=secret_provider
            )
        )

        self._engine = get_engine(dsn=self._sql_connection_string)

    def _init_tables(self):
        self._init_model_endpoints_table()
        self._init_application_results_table()
        self._init_monitoring_schedules_table()

    def _init_model_endpoints_table(self):
        self.ModelEndpointsTable = (
            mlrun.model_monitoring.db.stores.sqldb.models._get_model_endpoints_table(
                connection_string=self._sql_connection_string
            )
        )
        self._tables[
            mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_ENDPOINTS
        ] = self.ModelEndpointsTable

    def _init_application_results_table(self):
        self.ApplicationResultsTable = (
            mlrun.model_monitoring.db.stores.sqldb.models._get_application_result_table(
                connection_string=self._sql_connection_string
            )
        )
        self._tables[
            mlrun.common.schemas.model_monitoring.FileTargetKind.APP_RESULTS
        ] = self.ApplicationResultsTable

    def _init_monitoring_schedules_table(self):
        self.MonitoringSchedulesTable = mlrun.model_monitoring.db.stores.sqldb.models._get_monitoring_schedules_table(
            connection_string=self._sql_connection_string
        )
        self._tables[
            mlrun.common.schemas.model_monitoring.FileTargetKind.MONITORING_SCHEDULES
        ] = self.MonitoringSchedulesTable

    def _write(self, table: str, event: dict[str, typing.Any]):
        """
        Create a new record in the SQL table.

        :param table: Target table name.
        :param event: Event dictionary that will be written into the DB.
        """

        with self._engine.connect() as connection:
            # Convert the result into a pandas Dataframe and write it into the database
            event_df = pd.DataFrame([event])

            event_df.to_sql(table, con=connection, index=False, if_exists="append")

    def _update(
        self,
        attributes: dict[str, typing.Any],
        table: sqlalchemy.orm.decl_api.DeclarativeMeta,
        **filtered_values,
    ):
        """
        Update a record in the SQL table.

        :param attributes:  Dictionary of attributes that will be used for update the record. Note that the keys
                            of the attributes dictionary should exist in the SQL table.
        :param table:       SQLAlchemy declarative table.

        """
        filter_query_ = []
        for _filter in filtered_values:
            filter_query_.append(f"{_filter} = '{filtered_values[_filter]}'")

        with create_session(dsn=self._sql_connection_string) as session:
            # Generate and commit the update session query
            session.query(table).filter(sqlalchemy.sql.text(*filter_query_)).update(
                attributes, synchronize_session=False
            )
            session.commit()

    def _get(self, table: sqlalchemy.orm.decl_api.DeclarativeMeta, **filtered_values):
        """
        Get a record from the SQL table.

        param table: SQLAlchemy declarative table.
        """

        filter_query_ = []
        for _filter in filtered_values:
            filter_query_.append(f"{_filter} = '{filtered_values[_filter]}'")
        with create_session(dsn=self._sql_connection_string) as session:
            try:
                # Generate the get query
                return (
                    session.query(table)
                    .filter(sqlalchemy.sql.text(*filter_query_))
                    .one_or_none()
                )
            except sqlalchemy.exc.ProgrammingError:
                # Probably table doesn't exist, try to create tables
                self._create_tables_if_not_exist()
                return

    def _delete(
        self, table: sqlalchemy.orm.decl_api.DeclarativeMeta, **filtered_values
    ):
        """
        Delete records from the SQL table.

        param table: SQLAlchemy declarative table.
        """
        filter_query_ = []
        for _filter in filtered_values:
            filter_query_.append(f"{_filter} = '{filtered_values[_filter]}'")
        with create_session(dsn=self._sql_connection_string) as session:
            # Generate and commit the delete query
            session.query(table).filter(sqlalchemy.sql.text(*filter_query_)).delete(
                synchronize_session=False
            )
            session.commit()

    def write_model_endpoint(self, endpoint: dict[str, typing.Any]):
        """
        Create a new endpoint record in the SQL table. This method also creates the model endpoints table within the
        SQL database if not exist.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """

        # Adjust timestamps fields
        endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.FIRST_REQUEST] = (
            endpoint
        )[
            mlrun.common.schemas.model_monitoring.EventFieldType.LAST_REQUEST
        ] = mlrun.utils.datetime_now()

        self._write(
            table=mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_ENDPOINTS,
            event=endpoint,
        )

    def update_model_endpoint(
        self, endpoint_id: str, attributes: dict[str, typing.Any]
    ):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the SQL table.

        """
        self._init_model_endpoints_table()

        attributes.pop(
            mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID, None
        )

        filter_endpoint = {
            mlrun.common.schemas.model_monitoring.EventFieldType.UID: endpoint_id
        }

        self._update(
            attributes=attributes, table=self.ModelEndpointsTable, **filter_endpoint
        )

    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the SQL record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """
        self._init_model_endpoints_table()

        filter_endpoint = {
            mlrun.common.schemas.model_monitoring.EventFieldType.UID: endpoint_id
        }
        # Delete the model endpoint record using sqlalchemy ORM
        self._delete(table=self.ModelEndpointsTable, **filter_endpoint)

    def get_model_endpoint(
        self,
        endpoint_id: str,
    ) -> dict[str, typing.Any]:
        """
        Get a single model endpoint record.

        :param endpoint_id: The unique id of the model endpoint.

        :return: A model endpoint record as a dictionary.

        :raise MLRunNotFoundError: If the model endpoints table was not found or the model endpoint id was not found.
        """
        self._init_model_endpoints_table()

        # Get the model endpoint record using sqlalchemy ORM
        filter_endpoint = {
            mlrun.common.schemas.model_monitoring.EventFieldType.UID: endpoint_id
        }
        endpoint_record = self._get(table=self.ModelEndpointsTable, **filter_endpoint)

        if not endpoint_record:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Convert the database values and the table columns into a python dictionary
        return endpoint_record.to_dict()

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: list[str] = None,
        top_level: bool = None,
        uids: list = None,
    ) -> list[dict[str, typing.Any]]:
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
        self._init_model_endpoints_table()
        # Generate an empty model endpoints that will be filled afterwards with model endpoint dictionaries
        endpoint_list = []

        model_endpoints_table = (
            self.ModelEndpointsTable.__table__  # pyright: ignore[reportGeneralTypeIssues]
        )

        # Get the model endpoints records using sqlalchemy ORM
        with create_session(dsn=self._sql_connection_string) as session:
            # Generate the list query
            query = session.query(self.ModelEndpointsTable).filter_by(
                project=self.project
            )

            # Apply filters
            if model:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.MODEL,
                    filtered_values=[model],
                )
            if function:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mlrun.common.schemas.model_monitoring.EventFieldType.FUNCTION,
                    filtered_values=[function],
                )
            if uids:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
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
                    model_endpoints_table=model_endpoints_table,
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

    def write_application_result(self, event: dict[str, typing.Any]):
        """
        Write a new application result event in the target table.

        :param event: An event dictionary that represents the application result, should be corresponded to the
                      schema defined in the :py:class:`~mlrun.common.schemas.model_monitoring.constants.WriterEvent`
                      object.
        """
        self._init_application_results_table()

        application_filter_dict = {
            mlrun.common.schemas.model_monitoring.EventFieldType.UID: self._generate_application_result_uid(
                event
            )
        }

        application_record = self._get(
            table=self.ApplicationResultsTable, **application_filter_dict
        )
        if application_record:
            self._convert_to_datetime(
                event=event,
                key=mlrun.common.schemas.model_monitoring.WriterEvent.START_INFER_TIME,
            )
            self._convert_to_datetime(
                event=event,
                key=mlrun.common.schemas.model_monitoring.WriterEvent.END_INFER_TIME,
            )
            # Update an existing application result
            self._update(
                attributes=event,
                table=self.ApplicationResultsTable,
                **application_filter_dict,
            )
        else:
            # Write a new application result
            event[mlrun.common.schemas.model_monitoring.EventFieldType.UID] = (
                application_filter_dict[
                    mlrun.common.schemas.model_monitoring.EventFieldType.UID
                ]
            )

            self._write(
                table=mlrun.common.schemas.model_monitoring.FileTargetKind.APP_RESULTS,
                event=event,
            )

    @staticmethod
    def _convert_to_datetime(event: dict[str, typing.Any], key: str):
        if isinstance(event[key], str):
            event[key] = datetime.datetime.fromisoformat(event[key])

    @staticmethod
    def _generate_application_result_uid(event: dict[str, typing.Any]) -> str:
        return (
            event[mlrun.common.schemas.model_monitoring.WriterEvent.ENDPOINT_ID]
            + "_"
            + event[mlrun.common.schemas.model_monitoring.WriterEvent.APPLICATION_NAME]
            + "_"
            + event[mlrun.common.schemas.model_monitoring.WriterEvent.RESULT_NAME]
        )

    def get_last_analyzed(self, endpoint_id: str, application_name: str) -> int:
        """
        Get the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.

        :return: Timestamp as a Unix time.
        :raise:  MLRunNotFoundError if last analyzed value is not found.

        """
        self._init_monitoring_schedules_table()
        application_filter_dict = self.filter_endpoint_and_application_name(
            endpoint_id=endpoint_id, application_name=application_name
        )
        monitoring_schedule_record = self._get(
            table=self.MonitoringSchedulesTable, **application_filter_dict
        )
        if not monitoring_schedule_record:
            raise mlrun.errors.MLRunNotFoundError(
                f"No last analyzed value has been found for {application_name} "
                f"that processes model endpoint {endpoint_id}"
            )
        return monitoring_schedule_record.last_analyzed

    def update_last_analyzed(
        self, endpoint_id: str, application_name: str, last_analyzed: int
    ):
        """
        Update the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.
        :param last_analyzed:    Timestamp as a Unix time that represents the last analyzed time of a certain
                                 application and model endpoint.
        """
        self._init_monitoring_schedules_table()

        application_filter_dict = self.filter_endpoint_and_application_name(
            endpoint_id=endpoint_id, application_name=application_name
        )
        monitoring_schedule_record = self._get(
            table=self.MonitoringSchedulesTable, **application_filter_dict
        )
        if not monitoring_schedule_record:
            # Add a new record with empty last analyzed value
            self._write(
                table=mlrun.common.schemas.model_monitoring.FileTargetKind.MONITORING_SCHEDULES,
                event={
                    mlrun.common.schemas.model_monitoring.SchedulingKeys.UID: uuid.uuid4().hex,
                    mlrun.common.schemas.model_monitoring.SchedulingKeys.APPLICATION_NAME: application_name,
                    mlrun.common.schemas.model_monitoring.SchedulingKeys.ENDPOINT_ID: endpoint_id,
                    mlrun.common.schemas.model_monitoring.SchedulingKeys.LAST_ANALYZED: last_analyzed,
                },
            )

        self._update(
            attributes={
                mlrun.common.schemas.model_monitoring.SchedulingKeys.LAST_ANALYZED: last_analyzed
            },
            table=self.MonitoringSchedulesTable,
            **application_filter_dict,
        )

    def _delete_last_analyzed(self, endpoint_id: str = "", application_name: str = ""):
        self._init_monitoring_schedules_table()

        application_filter_dict = self.filter_endpoint_and_application_name(
            endpoint_id=endpoint_id, application_name=application_name
        )

        # Delete the model endpoint record using sqlalchemy ORM
        self._delete(table=self.MonitoringSchedulesTable, **application_filter_dict)

    def _delete_application_result(
        self, endpoint_id: str = "", application_name: str = ""
    ):
        self._init_application_results_table()

        application_filter_dict = self.filter_endpoint_and_application_name(
            endpoint_id=endpoint_id, application_name=application_name
        )

        # Delete the model endpoint record using sqlalchemy ORM
        self._delete(table=self.ApplicationResultsTable, **application_filter_dict)

    def _create_tables_if_not_exist(self):
        self._init_tables()

        for table in self._tables:
            # Create table if not exist. The `metadata` contains the `ModelEndpointsTable`
            if not self._engine.has_table(table):
                self._tables[table].metadata.create_all(  # pyright: ignore[reportGeneralTypeIssues]
                    bind=self._engine
                )

    @staticmethod
    def _filter_values(
        query: sqlalchemy.orm.query.Query,
        model_endpoints_table: sqlalchemy.Table,
        key_filter: str,
        filtered_values: list,
        combined=True,
    ) -> sqlalchemy.orm.query.Query:
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
        return query.filter(sqlalchemy.and_(*filter_query))

    @staticmethod
    def _validate_labels(
        endpoint_dict: dict,
        labels: list,
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

    @staticmethod
    def filter_endpoint_and_application_name(
        endpoint_id: str, application_name: str
    ) -> dict[str, str]:
        """Generate a dictionary filter for endpoint id and application name"""
        if not endpoint_id and not application_name:
            raise mlrun.errors.MLRunBadRequestError(
                "Please provide a valid endpoint_id and/or application_name"
            )
        application_filter_dict = {}
        if endpoint_id:
            application_filter_dict[
                mlrun.common.schemas.model_monitoring.SchedulingKeys.ENDPOINT_ID
            ] = endpoint_id
        if application_name:
            application_filter_dict[
                mlrun.common.schemas.model_monitoring.SchedulingKeys.APPLICATION_NAME
            ] = application_name
        return application_filter_dict

    def delete_model_endpoints_resources(self, endpoints: list[dict[str, typing.Any]]):
        """
        Delete all model endpoints resources in both SQL and the time series DB.

        :param endpoints: A list of model endpoints flattened dictionaries.
        """

        for endpoint_dict in endpoints:
            endpoint_id = endpoint_dict[
                mlrun.common.schemas.model_monitoring.EventFieldType.UID
            ]

            # Delete last analyzed records
            self._delete_last_analyzed(endpoint_id=endpoint_id)

            # Delete application results records
            self._delete_application_result(endpoint_id=endpoint_id)

            # Delete model endpoint record
            self.delete_model_endpoint(endpoint_id=endpoint_id)

    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> dict[str, list[tuple[str, float]]]:
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
