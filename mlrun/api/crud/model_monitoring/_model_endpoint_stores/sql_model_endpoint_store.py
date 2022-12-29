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
from sqlalchemy.orm import sessionmaker

import mlrun
import mlrun.api.schemas
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.utils import logger

from .model_endpoint_store import ModelEndpointStore


class _ModelEndpointSQLStore(ModelEndpointStore):

    """
    Handles the DB operations when the DB target is from type SQL. For the SQL operations, we use SQLAlchemy, a Python
    SQL toolkit that handles the communication with the database.  When using SQL for storing the model endpoints
    record, the user needs to provide a valid connection string for the database.
    """

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
        self.connection_string = connection_string
        self.table_name = model_monitoring_constants.EventFieldType.MODEL_ENDPOINTS

    def write_model_endpoint(self, endpoint: mlrun.api.schemas.ModelEndpoint):
        """
        Create a new endpoint record in the SQL table. This method also creates the model endpoints table within the
        SQL database if not exist.

        :param endpoint: ModelEndpoint object that will be written into the DB.
        """

        engine = db.create_engine(self.connection_string)

        with engine.connect():
            if not engine.has_table(self.table_name):
                logger.info("Creating new model endpoints table in DB")
                # Define schema and table for the model endpoints table as required by the SQL table structure
                metadata = db.MetaData()
                self._get_table(self.table_name, metadata)

                # Create the table that stored in the MetaData object (if not exist)
                metadata.create_all(engine)

            # Retrieving the relevant attributes from the model endpoint object
            endpoint_dict = self.get_params(endpoint=endpoint)

            # Convert the result into a pandas Dataframe and write it into the database
            endpoint_df = pd.DataFrame([endpoint_dict])
            endpoint_df.to_sql(
                self.table_name, con=engine, index=False, if_exists="append"
            )

    def update_model_endpoint(self, endpoint_id: str, attributes: typing.Dict):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the SQL table.

        """

        engine = db.create_engine(self.connection_string)
        with engine.connect():
            # Generate the sqlalchemy.schema.Table object that represents the model endpoints table
            metadata = db.MetaData()
            model_endpoints_table = db.Table(
                self.table_name, metadata, autoload=True, autoload_with=engine
            )

            # Define and execute the query with the given attributes and the related model endpoint id
            update_query = (
                db.update(model_endpoints_table)
                .values(attributes)
                .where(
                    model_endpoints_table.c[
                        model_monitoring_constants.EventFieldType.ENDPOINT_ID
                    ]
                    == endpoint_id
                )
            )
            engine.execute(update_query)

    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the SQL record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """
        engine = db.create_engine(self.connection_string)
        with engine.connect():
            # Generate the sqlalchemy.schema.Table object that represents the model endpoints table
            metadata = db.MetaData()
            model_endpoints_table = db.Table(
                self.table_name, metadata, autoload=True, autoload_with=engine
            )

            # Delete the model endpoint record using sqlalchemy ORM
            session = sessionmaker(bind=engine)()

            session.query(model_endpoints_table).filter_by(
                endpoint_id=endpoint_id
            ).delete()
            session.commit()
            session.close()

    def get_model_endpoint(
        self,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
        endpoint_id: str = None,
        convert_to_endpoint_object: bool = True,
    ) -> typing.Union[mlrun.api.schemas.ModelEndpoint, dict]:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
        result.

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
        :param endpoint_id:                The unique id of the model endpoint.
        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a ModelEndpoint or not. True by default.

        :return: A ModelEndpoint object or a model endpoint dictionary if convert_to_endpoint_object is False.
        """
        logger.info(
            "Getting model endpoint record from SQL",
            endpoint_id=endpoint_id,
        )

        engine = db.create_engine(self.connection_string)

        # Validate that the model endpoints table exists
        if not engine.has_table(self.table_name):
            raise mlrun.errors.MLRunNotFoundError(f"Table {self.table_name} not found")

        with engine.connect():

            # Generate the sqlalchemy.schema.Table object that represents the model endpoints table
            metadata = db.MetaData()
            model_endpoints_table = db.Table(
                self.table_name, metadata, autoload=True, autoload_with=engine
            )

            # Get the model endpoint record using sqlalchemy ORM
            session = sessionmaker(bind=engine)()

            columns = model_endpoints_table.columns.keys()
            values = (
                session.query(model_endpoints_table)
                .filter_by(endpoint_id=endpoint_id)
                .filter_by()
                .all()
            )
            session.close()

        if len(values) == 0:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Convert the database values and the table columns into a python dictionary
        endpoint = dict(zip(columns, values[0]))

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
        labels: typing.Union[typing.List[str], typing.Dict] = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of ModelEndpoint objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available ModelEndpoint objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:         A list of real-time metrics to return for each model endpoint. There are pre-defined
                                real-time metrics for model endpoints such as predictions_per_second and latency_avg_5m
                                but also custom metrics defined by the user. Please note that these metrics are stored
                                in the time series DB and the results will be appeared under
                                model_endpoint.spec.metrics.
        :param start:           The start time of the metrics. Can be represented by a string containing an RFC 3339
                                time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for
                                 the earliest time.
        :param uids:             List of model endpoint unique ids to include in the result.

        :return: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """

        engine = db.create_engine(self.connection_string)

        # Generate an empty ModelEndpointList that will be filled afterwards with ModelEndpoint objects
        endpoint_list = mlrun.api.schemas.model_endpoints.ModelEndpointList(
            endpoints=[]
        )

        # Validate that the model endpoints table exists
        if not engine.has_table(self.table_name):
            logger.warn(
                "Table not found, return an empty ModelEndpointList",
                table=self.table_name,
            )
            return endpoint_list

        with engine.connect():
            # Generate the sqlalchemy.schema.Table object that represents the model endpoints table
            metadata = db.MetaData()
            model_endpoints_table = db.Table(
                self.table_name, metadata, autoload=True, autoload_with=engine
            )

            # Get the model endpoints records using sqlalchemy ORM
            session = sessionmaker(bind=engine)()

            columns = model_endpoints_table.columns.keys()
            query = session.query(model_endpoints_table).filter_by(project=self.project)

            # Apply filters
            if model:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.MODEL,
                    filtered_values=[model],
                )
            if function:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.FUNCTION,
                    filtered_values=[function],
                )
            if uids:
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.ENDPOINT_ID,
                    filtered_values=uids,
                    combined=False,
                )
            if top_level:
                node_ep = str(mlrun.utils.model_monitoring.EndpointType.NODE_EP.value)
                router_ep = str(mlrun.utils.model_monitoring.EndpointType.ROUTER.value)
                endpoint_types = [node_ep, router_ep]
                query = self._filter_values(
                    query=query,
                    model_endpoints_table=model_endpoints_table,
                    key_filter=model_monitoring_constants.EventFieldType.ENDPOINT_TYPE,
                    filtered_values=endpoint_types,
                    combined=False,
                )
            # Labels from type list won't be supported from 1.4.0
            # TODO: Remove in 1.4.0
            if labels and isinstance(labels, typing.List):
                logger.warn(
                    "Labels should be from type dictionary, not list",
                    labels=labels,
                )

            # Convert the results from the DB into a ModelEndpoint object and append it to the ModelEndpointList
            for endpoint_values in query.all():
                endpoint_dict = dict(zip(columns, endpoint_values))

                # Filter labels
                if labels and labels != json.loads(
                    endpoint_dict.get(model_monitoring_constants.EventFieldType.LABELS)
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
            session.close()
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

        # Generating a tuple with the relevant filters
        filter_query = ()
        for _filter in filtered_values:
            filter_query += (model_endpoints_table.c[key_filter] == _filter,)

        # Apply AND/OR operator on the SQL query object with the filters tuple
        if combined:
            return query.filter(db.and_(*filter_query))
        else:
            return query.filter(db.or_(*filter_query))

    @staticmethod
    def _get_table(table_name: str, metadata: db.MetaData):
        """Declaring a new SQL table object with the required model endpoints columns. Below you can find the list
        of available columns:

        [endpoint_id, state, project, function_uri, model, model_class, labels, model_uri, stream_path, active,
        monitoring_mode, feature_stats, current_stats, feature_names, children, label_names, timestamp, endpoint_type,
        children_uids, drift_measures, drift_status, monitor_configuration, monitoring_feature_set_uri, first_request,
        last_request, error_count, metrics]

        :param table_name: Model endpoints SQL table name.
        :param metadata:   SQLAlchemy MetaData object that used to describe the SQL DataBase. The below method uses the
                           MetaData object for declaring a table.
        """

        db.Table(
            table_name,
            metadata,
            db.Column(
                model_monitoring_constants.EventFieldType.ENDPOINT_ID,
                db.String(40),
                primary_key=True,
            ),
            db.Column(model_monitoring_constants.EventFieldType.STATE, db.String(10)),
            db.Column(model_monitoring_constants.EventFieldType.PROJECT, db.String(40)),
            db.Column(
                model_monitoring_constants.EventFieldType.FUNCTION_URI,
                db.String(255),
            ),
            db.Column(model_monitoring_constants.EventFieldType.MODEL, db.String(255)),
            db.Column(
                model_monitoring_constants.EventFieldType.MODEL_CLASS,
                db.String(255),
            ),
            db.Column(model_monitoring_constants.EventFieldType.LABELS, db.Text),
            db.Column(
                model_monitoring_constants.EventFieldType.MODEL_URI, db.String(255)
            ),
            db.Column(model_monitoring_constants.EventFieldType.STREAM_PATH, db.Text),
            db.Column(
                model_monitoring_constants.EventFieldType.ALGORITHM,
                db.String(255),
            ),
            db.Column(model_monitoring_constants.EventFieldType.ACTIVE, db.Boolean),
            db.Column(
                model_monitoring_constants.EventFieldType.MONITORING_MODE,
                db.String(10),
            ),
            db.Column(model_monitoring_constants.EventFieldType.FEATURE_STATS, db.Text),
            db.Column(model_monitoring_constants.EventFieldType.CURRENT_STATS, db.Text),
            db.Column(model_monitoring_constants.EventFieldType.FEATURE_NAMES, db.Text),
            db.Column(model_monitoring_constants.EventFieldType.CHILDREN, db.Text),
            db.Column(model_monitoring_constants.EventFieldType.LABEL_NAMES, db.Text),
            db.Column(model_monitoring_constants.EventFieldType.TIMESTAMP, db.DateTime),
            db.Column(
                model_monitoring_constants.EventFieldType.ENDPOINT_TYPE,
                db.String(10),
            ),
            db.Column(model_monitoring_constants.EventFieldType.CHILDREN_UIDS, db.Text),
            db.Column(
                model_monitoring_constants.EventFieldType.DRIFT_MEASURES, db.Text
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.DRIFT_STATUS,
                db.String(40),
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.MONITOR_CONFIGURATION,
                db.Text,
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.FEATURE_SET_URI,
                db.String(255),
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.FIRST_REQUEST,
                db.String(40),
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.LAST_REQUEST,
                db.String(40),
            ),
            db.Column(
                model_monitoring_constants.EventFieldType.ERROR_COUNT, db.Integer
            ),
            db.Column(model_monitoring_constants.EventFieldType.METRICS, db.Text),
        )

    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        """
        Delete all model endpoints resources in both SQL and the time series DB. In addition, delete the model
        endpoints table from SQL if it's empty at the end of the process.

        :param endpoints: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """

        # Delete model endpoint record from SQL table
        for endpoint in endpoints.endpoints:
            self.delete_model_endpoint(
                endpoint.metadata.uid,
            )

        # Drop the SQL table if it's empty
        self._drop_table()

    def _drop_table(self):
        """Delete model endpoints SQL table. If table is not empty, then it won't be deleted."""
        engine = db.create_engine(self.connection_string)
        with engine.connect():
            if not engine.has_table(self.table_name):
                logger.warn(
                    "Table not found",
                    table=self.table_name,
                )
                return

            # Generate the sqlalchemy.schema.Table object that represents the model endpoints table
            metadata = db.MetaData()
            model_endpoints_table = db.Table(
                self.table_name, metadata, autoload=True, autoload_with=engine
            )

            # Count the model endpoint records using sqlalchemy ORM
            session = sessionmaker(bind=engine)()
            rows = session.query(model_endpoints_table).count()
            session.close()

            # Drop the table if no records have been found
            if rows > 0:
                logger.info(
                    "Table is not empty and therefore won't be deleted from DB",
                    table_name=self.table_name,
                )
            else:
                metadata.drop_all(bind=engine, tables=[model_endpoints_table])
                logger.info(
                    "Table has been deleted from SQL", table_name=self.table_name
                )
        return
