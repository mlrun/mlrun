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

    def _init_model_endpoints_table(self):
        self.model_endpoints_table = (
            mlrun.model_monitoring.db.stores.sqldb.models._get_model_endpoints_table(
                connection_string=self._sql_connection_string
            )
        )
        self._tables[mm_schemas.EventFieldType.MODEL_ENDPOINTS] = (
            self.model_endpoints_table
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
        model: typing.Optional[str] = None,
        function: typing.Optional[str] = None,
        labels: typing.Optional[list[str]] = None,
        top_level: typing.Optional[bool] = None,
        uids: typing.Optional[list] = None,
        include_stats: typing.Optional[bool] = None,
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

    @staticmethod
    def _convert_to_datetime(event: dict[str, typing.Any], key: str) -> None:
        if isinstance(event[key], str):
            event[key] = datetime.datetime.fromisoformat(event[key])
        event[key] = event[key].astimezone(tz=datetime.timezone.utc)

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
