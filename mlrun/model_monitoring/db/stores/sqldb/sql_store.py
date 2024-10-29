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
import typing
import uuid

import pandas as pd
import sqlalchemy
import sqlalchemy.exc
import sqlalchemy.orm
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.sql.elements import BinaryExpression

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.stores.sqldb.models
import mlrun.model_monitoring.helpers
from mlrun.common.db.sql_session import create_session, get_engine
from mlrun.model_monitoring.db import StoreBase
from mlrun.utils import datetime_now, logger


class SQLStoreBase(StoreBase):
    type: typing.ClassVar[str] = mm_schemas.ModelEndpointTarget.SQL
    """
    Handles the DB operations when the DB target is from type SQL. For the SQL operations, we use SQLAlchemy, a Python
    SQL toolkit that handles the communication with the database.  When using SQL for storing the model monitoring
    data, the user needs to provide a valid connection string for the database.
    """

    _tables = {}

    def __init__(
        self,
        project: str,
        **kwargs,
    ):
        """
        Initialize SQL store target object.

        :param project:               The name of the project.
        """

        super().__init__(project=project)

        if "store_connection_string" not in kwargs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "connection_string is a required parameter for SQLStoreBase."
            )

        self._sql_connection_string = kwargs.get("store_connection_string")
        self._engine = None
        self._init_tables()

    @property
    def engine(self) -> Engine:
        if not self._engine:
            self._engine = get_engine(dsn=self._sql_connection_string)
        return self._engine

    def create_tables(self):
        self._create_tables_if_not_exist()

    def _init_tables(self):
        self._init_model_endpoints_table()
        self._init_application_results_table()
        self._init_application_metrics_table()
        self._init_monitoring_schedules_table()

    def _init_model_endpoints_table(self):
        self.model_endpoints_table = (
            mlrun.model_monitoring.db.stores.sqldb.models._get_model_endpoints_table(
                connection_string=self._sql_connection_string
            )
        )
        self._tables[mm_schemas.EventFieldType.MODEL_ENDPOINTS] = (
            self.model_endpoints_table
        )

    def _init_application_results_table(self):
        self.application_results_table = (
            mlrun.model_monitoring.db.stores.sqldb.models._get_application_result_table(
                connection_string=self._sql_connection_string
            )
        )
        self._tables[mm_schemas.FileTargetKind.APP_RESULTS] = (
            self.application_results_table
        )

    def _init_application_metrics_table(self) -> None:
        self.application_metrics_table = mlrun.model_monitoring.db.stores.sqldb.models._get_application_metrics_table(
            connection_string=self._sql_connection_string
        )
        self._tables[mm_schemas.FileTargetKind.APP_METRICS] = (
            self.application_metrics_table
        )

    def _init_monitoring_schedules_table(self):
        self.MonitoringSchedulesTable = mlrun.model_monitoring.db.stores.sqldb.models._get_monitoring_schedules_table(
            connection_string=self._sql_connection_string
        )
        self._tables[mm_schemas.FileTargetKind.MONITORING_SCHEDULES] = (
            self.MonitoringSchedulesTable
        )

    def _write(self, table_name: str, event: dict[str, typing.Any]) -> None:
        """
        Create a new record in the SQL table.

        :param table_name: Target table name.
        :param event:      Event dictionary that will be written into the DB.
        """
        with self.engine.connect() as connection:
            # Convert the result into a pandas Dataframe and write it into the database
            event_df = pd.DataFrame([event])
            event_df.to_sql(table_name, con=connection, index=False, if_exists="append")

    def _update(
        self,
        attributes: dict[str, typing.Any],
        table: sqlalchemy.orm.decl_api.DeclarativeMeta,
        criteria: list[BinaryExpression],
    ) -> None:
        """
        Update a record in the SQL table.

        :param attributes:  Dictionary of attributes that will be used for update the record. Note that the keys
                            of the attributes dictionary should exist in the SQL table.
        :param table:       SQLAlchemy declarative table.
        :param criteria:    A list of binary expressions that filter the query.
        """
        with create_session(dsn=self._sql_connection_string) as session:
            # Generate and commit the update session query
            session.query(
                table  # pyright: ignore[reportOptionalCall]
            ).filter(*criteria).update(attributes, synchronize_session=False)
            session.commit()

    def _get(
        self,
        table: sqlalchemy.orm.decl_api.DeclarativeMeta,
        criteria: list[BinaryExpression],
    ):
        """
        Get a record from the SQL table.

        param table:     SQLAlchemy declarative table.
        :param criteria: A list of binary expressions that filter the query.
        """
        with create_session(dsn=self._sql_connection_string) as session:
            logger.debug(
                "Querying the DB",
                table=table.__name__,
                criteria=[str(criterion) for criterion in criteria],
            )
            # Generate the get query
            return (
                session.query(table)  # pyright: ignore[reportOptionalCall]
                .filter(*criteria)
                .one_or_none()
            )

    def _delete(
        self,
        table: sqlalchemy.orm.decl_api.DeclarativeMeta,
        criteria: list[BinaryExpression],
    ) -> None:
        """
        Delete records from the SQL table.

        param table:     SQLAlchemy declarative table.
        :param criteria: A list of binary expressions that filter the query.
        """
        if not self.engine.has_table(table.__tablename__):
            logger.debug(
                f"Table {table.__tablename__} does not exist in the database. Skipping deletion."
            )
            return
        with create_session(dsn=self._sql_connection_string) as session:
            # Generate and commit the delete query
            session.query(
                table  # pyright: ignore[reportOptionalCall]
            ).filter(*criteria).delete(synchronize_session=False)
            session.commit()

    def write_model_endpoint(self, endpoint: dict[str, typing.Any]):
        """
        Create a new endpoint record in the SQL table. This method also creates the model endpoints table within the
        SQL database if not exist.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """

        # Adjust timestamps fields
        endpoint[mm_schemas.EventFieldType.FIRST_REQUEST] = (endpoint)[
            mm_schemas.EventFieldType.LAST_REQUEST
        ] = datetime_now()

        self._write(
            table_name=mm_schemas.EventFieldType.MODEL_ENDPOINTS, event=endpoint
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

        attributes.pop(mm_schemas.EventFieldType.ENDPOINT_ID, None)

        self._update(
            attributes=attributes,
            table=self.model_endpoints_table,
            criteria=[self.model_endpoints_table.uid == endpoint_id],
        )

    def delete_model_endpoint(self, endpoint_id: str) -> None:
        """
        Deletes the SQL record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """
        # Delete the model endpoint record using sqlalchemy ORM
        self._delete(
            table=self.model_endpoints_table,
            criteria=[self.model_endpoints_table.uid == endpoint_id],
        )

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

        # Get the model endpoint record
        endpoint_record = self._get(
            table=self.model_endpoints_table,
            criteria=[self.model_endpoints_table.uid == endpoint_id],
        )

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
        include_stats: bool = None,
    ) -> list[dict[str, typing.Any]]:
        # Generate an empty model endpoints that will be filled afterwards with model endpoint dictionaries
        endpoint_list = []

        model_endpoints_table = (
            self.model_endpoints_table.__table__  # pyright: ignore[reportAttributeAccessIssue]
        )
        # Get the model endpoints records using sqlalchemy ORM
        with create_session(dsn=self._sql_connection_string) as session:
            # Generate the list query
            query = session.query(self.model_endpoints_table).filter_by(
                project=self.project
            )

            # Apply filters
            if model:
                model = model if ":" in model else f"{model}:latest"
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mm_schemas.EventFieldType.MODEL,
                    filtered_values=[model],
                )
            if function:
                function_uri = f"{self.project}/{function}"
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mm_schemas.EventFieldType.FUNCTION_URI,
                    filtered_values=[function_uri],
                )
            if uids:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mm_schemas.EventFieldType.UID,
                    filtered_values=uids,
                    combined=False,
                )
            if top_level:
                node_ep = str(mm_schemas.EndpointType.NODE_EP.value)
                router_ep = str(mm_schemas.EndpointType.ROUTER.value)
                endpoint_types = [node_ep, router_ep]
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=mm_schemas.EventFieldType.ENDPOINT_TYPE,
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

                if not include_stats:
                    # Exclude these fields when listing model endpoints to avoid returning too much data (ML-6594)
                    # TODO: Remove stats from table schema (ML-7196)
                    endpoint_dict.pop(mm_schemas.EventFieldType.FEATURE_STATS)
                    endpoint_dict.pop(mm_schemas.EventFieldType.CURRENT_STATS)

                endpoint_list.append(endpoint_dict)

        return endpoint_list

    def write_application_event(
        self,
        event: dict[str, typing.Any],
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a new application event in the target table.

        :param event: An event dictionary that represents the application result or metric,
                      should be corresponded to the schema defined in the
                      :py:class:`~mm_constants.constants.WriterEvent` object.
        :param kind: The type of the event, can be either "result" or "metric".
        """

        if kind == mm_schemas.WriterEventKind.METRIC:
            table = self.application_metrics_table
            table_name = mm_schemas.FileTargetKind.APP_METRICS
        elif kind == mm_schemas.WriterEventKind.RESULT:
            table = self.application_results_table
            table_name = mm_schemas.FileTargetKind.APP_RESULTS
        else:
            raise ValueError(f"Invalid {kind = }")

        application_result_uid = self._generate_application_result_uid(event, kind=kind)
        criteria = [table.uid == application_result_uid]

        application_record = self._get(table=table, criteria=criteria)
        if application_record:
            self._convert_to_datetime(
                event=event, key=mm_schemas.WriterEvent.START_INFER_TIME
            )
            self._convert_to_datetime(
                event=event, key=mm_schemas.WriterEvent.END_INFER_TIME
            )
            # Update an existing application result
            self._update(attributes=event, table=table, criteria=criteria)
        else:
            # Write a new application result
            event[mm_schemas.EventFieldType.UID] = application_result_uid
            self._write(table_name=table_name, event=event)

    @staticmethod
    def _convert_to_datetime(event: dict[str, typing.Any], key: str) -> None:
        if isinstance(event[key], str):
            event[key] = datetime.datetime.fromisoformat(event[key])
        event[key] = event[key].astimezone(tz=datetime.timezone.utc)

    @staticmethod
    def _generate_application_result_uid(
        event: dict[str, typing.Any],
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> str:
        if kind == mm_schemas.WriterEventKind.RESULT:
            name = event[mm_schemas.ResultData.RESULT_NAME]
        else:
            name = event[mm_schemas.MetricData.METRIC_NAME]
        return "_".join(
            [
                event[mm_schemas.WriterEvent.ENDPOINT_ID],
                event[mm_schemas.WriterEvent.APPLICATION_NAME],
                name,
            ]
        )

    @staticmethod
    def _get_filter_criteria(
        *,
        table: sqlalchemy.orm.decl_api.DeclarativeMeta,
        endpoint_id: str,
        application_name: typing.Optional[str] = None,
    ) -> list[BinaryExpression]:
        """
        Return the filter criteria for the given endpoint_id and application_name.
        Note: the table object must include the relevant columns:
        `endpoint_id` and `application_name`.
        """
        criteria = [table.endpoint_id == endpoint_id]
        if application_name is not None:
            criteria.append(table.application_name == application_name)
        return criteria

    def get_last_analyzed(self, endpoint_id: str, application_name: str) -> int:
        """
        Get the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.

        :return: Timestamp as a Unix time.
        :raise:  MLRunNotFoundError if last analyzed value is not found.
        """
        monitoring_schedule_record = self._get(
            table=self.MonitoringSchedulesTable,
            criteria=self._get_filter_criteria(
                table=self.MonitoringSchedulesTable,
                endpoint_id=endpoint_id,
                application_name=application_name,
            ),
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
        criteria = self._get_filter_criteria(
            table=self.MonitoringSchedulesTable,
            endpoint_id=endpoint_id,
            application_name=application_name,
        )
        monitoring_schedule_record = self._get(
            table=self.MonitoringSchedulesTable, criteria=criteria
        )
        if not monitoring_schedule_record:
            # Add a new record with last analyzed value
            self._write(
                table_name=mm_schemas.FileTargetKind.MONITORING_SCHEDULES,
                event={
                    mm_schemas.SchedulingKeys.UID: uuid.uuid4().hex,
                    mm_schemas.SchedulingKeys.APPLICATION_NAME: application_name,
                    mm_schemas.SchedulingKeys.ENDPOINT_ID: endpoint_id,
                    mm_schemas.SchedulingKeys.LAST_ANALYZED: last_analyzed,
                },
            )

        self._update(
            attributes={mm_schemas.SchedulingKeys.LAST_ANALYZED: last_analyzed},
            table=self.MonitoringSchedulesTable,
            criteria=criteria,
        )

    def _delete_last_analyzed(
        self, endpoint_id: str, application_name: typing.Optional[str] = None
    ) -> None:
        criteria = self._get_filter_criteria(
            table=self.MonitoringSchedulesTable,
            endpoint_id=endpoint_id,
            application_name=application_name,
        )
        # Delete the model endpoint record using sqlalchemy ORM
        self._delete(table=self.MonitoringSchedulesTable, criteria=criteria)

    def _delete_application_result(
        self, endpoint_id: str, application_name: typing.Optional[str] = None
    ) -> None:
        criteria = self._get_filter_criteria(
            table=self.application_results_table,
            endpoint_id=endpoint_id,
            application_name=application_name,
        )
        # Delete the relevant records from the results table
        self._delete(table=self.application_results_table, criteria=criteria)

    def _delete_application_metrics(
        self, endpoint_id: str, application_name: typing.Optional[str] = None
    ) -> None:
        criteria = self._get_filter_criteria(
            table=self.application_metrics_table,
            endpoint_id=endpoint_id,
            application_name=application_name,
        )
        # Delete the relevant records from the metrics table
        self._delete(table=self.application_metrics_table, criteria=criteria)

    def _create_tables_if_not_exist(self):
        self._init_tables()

        for table in self._tables:
            # Create table if not exist. The `metadata` contains the `ModelEndpointsTable`
            db_name = make_url(self._sql_connection_string).database
            if not self.engine.has_table(table):
                logger.info(f"Creating table {table} on {db_name} db.")
                self._tables[table].metadata.create_all(bind=self.engine)
            else:
                logger.info(f"Table {table} already exists on {db_name} db.")

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

    def delete_model_endpoints_resources(self) -> None:
        """
        Delete all the model monitoring resources of the project in the SQL tables.
        """
        logger.debug(
            "Deleting model monitoring endpoints resources from the SQL tables",
            project=self.project,
        )
        endpoints = self.list_model_endpoints()

        for endpoint_dict in endpoints:
            endpoint_id = endpoint_dict[mm_schemas.EventFieldType.UID]
            logger.debug(
                "Deleting model endpoint resources from the SQL tables",
                endpoint_id=endpoint_id,
                project=self.project,
            )
            # Delete last analyzed records
            self._delete_last_analyzed(endpoint_id=endpoint_id)

            # Delete application results and metrics records
            self._delete_application_result(endpoint_id=endpoint_id)
            self._delete_application_metrics(endpoint_id=endpoint_id)

            # Delete model endpoint record
            self.delete_model_endpoint(endpoint_id=endpoint_id)
            logger.debug(
                "Successfully deleted model endpoint resources",
                endpoint_id=endpoint_id,
                project=self.project,
            )

        logger.debug(
            "Successfully deleted model monitoring endpoints resources from the SQL tables",
            project=self.project,
        )

    def get_model_endpoint_metrics(
        self, endpoint_id: str, type: mm_schemas.ModelEndpointMonitoringMetricType
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        """
        Fetch the model endpoint metrics or results (according to `type`) for the
        requested endpoint.
        """
        logger.debug(
            "Fetching metrics for model endpoint",
            project=self.project,
            endpoint_id=endpoint_id,
            type=type,
        )
        if type == mm_schemas.ModelEndpointMonitoringMetricType.METRIC:
            table = self.application_metrics_table
            name_col = mm_schemas.MetricData.METRIC_NAME
        else:
            table = self.application_results_table
            name_col = mm_schemas.ResultData.RESULT_NAME

        # Note: the block below does not use self._get, as we need here all the
        # results, not only `one_or_none`.
        with sqlalchemy.orm.Session(self.engine) as session:
            metric_rows = (
                session.query(table)  # pyright: ignore[reportOptionalCall]
                .filter(table.endpoint_id == endpoint_id)
                .all()
            )

        return [
            mm_schemas.ModelEndpointMonitoringMetric(
                project=self.project,
                app=metric_row.application_name,
                type=type,
                name=getattr(metric_row, name_col),
                full_name=mlrun.model_monitoring.helpers._compose_full_name(
                    project=self.project,
                    app=metric_row.application_name,
                    type=type,
                    name=getattr(metric_row, name_col),
                ),
            )
            for metric_row in metric_rows
        ]
