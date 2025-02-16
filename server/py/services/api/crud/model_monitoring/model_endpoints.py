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
import uuid
from datetime import datetime
from typing import Callable, Optional

import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.artifacts
import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.datastore
import mlrun.datastore.datastore_profile
import mlrun.errors
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
from mlrun.utils import logger, parse_artifact_uri

import framework.api.utils
import framework.utils.singletons.db
import services.api.crud.model_monitoring.deployment
import services.api.crud.model_monitoring.helpers
import services.api.crud.secrets

DEFAULT_FUNCTION_TAG = "latest"
ARCHIVE_LIMITATION = 5


class ModelEndpoints:
    """Provide different methods for handling model endpoints such as listing, writing and deleting"""

    async def create_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        creation_strategy: mlrun.common.schemas.ModelEndpointCreationStrategy,
        model_path: Optional[str] = None,
        upsert: bool = True,
    ) -> typing.Union[tuple[mlrun.common.schemas.ModelEndpoint, str, list[str], dict],]:
        """
        Creates model endpoint record in DB. The DB store target is defined either by a provided connection string
        or by the default store target that is defined in MLRun configuration.

        :param db_session:             A session that manages the current dialog with the database.
        :param model_endpoint:         Model endpoint object to update.
        :param creation_strategy: Strategy for creating or updating the model endpoint:
            * **overwrite**:
            1. If model endpoints with the same name exist, delete the `latest` one.
            2. Create a new model endpoint entry and set it as `latest`.
            * **inplace** (default):
            1. If model endpoints with the same name exist, update the `latest` entry.
            2. Otherwise, create a new entry.
            * **archive**:
            1. If model endpoints with the same name exist, preserve them.
            2. Create a new model endpoint with the same name and set it to `latest`.
        :param model_path:             The path to the model artifact.
        :param upsert:                 If True, will execute the creation/deletion/updating
                                       of the model endpoint in the DB.

        :return:    The created `ModelEndpoint` object, the method that was used to create it, the uids of the model
                    endpoints that were deleted and the attributes that were updated.
        :raise:     MLRunInvalidArgumentError if the creation strategy is not valid
        """
        if model_endpoint.spec.function_name and not model_endpoint.spec.function_tag:
            logger.info("Function tag not provided, setting to 'latest'")
            model_endpoint.spec.function_tag = DEFAULT_FUNCTION_TAG

        logger.info(
            "Creating Model Endpoint record",
            model_endpoint_metadata=model_endpoint.metadata,
            creation_strategy=creation_strategy,
        )

        if not model_endpoint.metadata.uid:
            model_endpoint.metadata.uid = uuid.uuid4().hex

        if not model_endpoint.spec.function_uid:
            # get function_uid from db
            try:
                logger.info("Getting function uid from db")
                current_function = await run_in_threadpool(
                    framework.utils.singletons.db.get_db().get_function,
                    db_session,
                    name=model_endpoint.spec.function_name,
                    tag=model_endpoint.spec.function_tag,
                    project=model_endpoint.metadata.project,
                )
                model_endpoint.spec.function_uid = current_function.get(
                    "metadata", {}
                ).get("uid")
            except mlrun.errors.MLRunNotFoundError:
                logger.info("The model endpoint is created on a non-existing function")
        model_obj = None
        if model_path and mlrun.datastore.is_store_uri(model_path):
            _, model_uri = mlrun.datastore.parse_store_uri(model_path)
            project, key, iteration, tag, tree, uid = parse_artifact_uri(
                model_uri, model_endpoint.metadata.project
            )
        else:
            project, key, iteration, tag, tree, uid = (
                model_endpoint.metadata.project,
                model_endpoint.spec.model_db_key,
                None,
                model_endpoint.spec.model_tag,
                None,
                model_endpoint.spec.model_uid,
            )
        try:
            logger.info("Getting model object from db")
            model_obj = mlrun.artifacts.dict_to_artifact(
                services.api.crud.Artifacts().get_artifact(
                    db_session,
                    key=key,
                    tag=tag,
                    iter=iteration,
                    project=project,
                    producer_id=tree,
                    object_uid=uid,
                )
            )

            model_endpoint.spec.model_name = model_obj.metadata.key
            model_endpoint.spec.model_db_key = model_obj.spec.db_key
            model_endpoint.spec.model_uid = model_obj.metadata.uid
            model_endpoint.spec.model_tag = model_obj.tag
            model_endpoint.metadata.labels.update(
                model_obj.labels
            )  # todo : check if we still need this
        except mlrun.errors.MLRunNotFoundError:
            logger.info("The model endpoint is created on a non-existing model")

        if (
            creation_strategy
            == mlrun.common.schemas.ModelEndpointCreationStrategy.INPLACE
        ):
            (
                model_endpoint,
                method,
                uid_to_delete,
                attributes,
            ) = await self._inplace_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                model_obj=model_obj,
                upsert=upsert,
            )
        elif (
            creation_strategy
            == mlrun.common.schemas.ModelEndpointCreationStrategy.OVERWRITE
        ):
            (
                model_endpoint,
                method,
                uid_to_delete,
                attributes,
            ) = await self._overwrite_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                model_obj=model_obj,
                upsert=upsert,
            )
        elif (
            creation_strategy
            == mlrun.common.schemas.ModelEndpointCreationStrategy.ARCHIVE
        ):
            (
                model_endpoint,
                method,
                uid_to_delete,
                attributes,
            ) = await self._archive_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                model_obj=model_obj,
                delete_old=True,
                upsert=upsert,
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{creation_strategy} is invalid creation strategy"
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system
        logger.info("Model endpoint created", endpoint_id=model_endpoint.metadata.uid)
        return model_endpoint, method, uid_to_delete, attributes

    async def create_model_endpoints(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoints_instructions: list[
            tuple[
                mlrun.common.schemas.ModelEndpoint,
                mm_constants.ModelEndpointCreationStrategy,
                str,
            ]
        ],
        project: str,
    ) -> None:
        # extra improvement to list all the relevant meps before - can be relevant to inplace and to the deletion
        # extra improvement to upsert all feature sets together
        # batch json creation
        model_endpoints_dict = {"create": [], "update": {}, "delete": []}
        for (
            model_endpoint,
            creation_strategy,
            model_path,
        ) in model_endpoints_instructions:
            (
                model_endpoint,
                method,
                uid_to_delete,
                attributes,
            ) = await self.create_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                creation_strategy=creation_strategy,
                model_path=model_path,
                upsert=False,
            )
            if method == "create":
                model_endpoints_dict.get(method).append(model_endpoint)
            elif method == "update":
                model_endpoints_dict.get(method)[model_endpoint.metadata.uid] = (
                    attributes
                )
            model_endpoints_dict.get("delete").extend(uid_to_delete)

        if model_endpoints_dict.get("create"):
            await run_in_threadpool(
                framework.utils.singletons.db.get_db().store_model_endpoints,
                session=db_session,
                project=project,
                model_endpoints=model_endpoints_dict.get("create"),
            )
        if model_endpoints_dict.get("update"):
            await run_in_threadpool(
                framework.utils.singletons.db.get_db().update_model_endpoints,
                session=db_session,
                project=project,
                attributes=model_endpoints_dict.get("update"),
            )

        if model_endpoints_dict.get("delete"):
            # delete old versions
            await run_in_threadpool(
                framework.utils.singletons.db.get_db().delete_model_endpoints,
                session=db_session,
                project=project,
                uids=model_endpoints_dict.get("delete"),
            )
            await run_in_threadpool(
                self._delete_model_endpoint_monitoring_infra,
                uids=model_endpoints_dict.get("delete"),
                project=project,
            )

    async def _inplace_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        model_obj: Optional[mlrun.artifacts.ModelArtifact] = None,
        upsert: bool = True,
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, str, list[str], dict]:
        try:
            logger.info("Getting model endpoint from db")
            exist_model_endpoint = await run_in_threadpool(
                framework.utils.singletons.db.get_db().get_model_endpoint,
                session=db_session,
                project=model_endpoint.metadata.project,
                name=model_endpoint.metadata.name,
                function_name=model_endpoint.spec.function_name,
                function_tag=model_endpoint.spec.function_tag,
            )
        except mlrun.errors.MLRunNotFoundError:
            exist_model_endpoint = None

        if not exist_model_endpoint:
            # there is no model endpoint with the same name
            # create a new model endpoint using the same logic as archive
            return await self._archive_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                upsert=upsert,
                model_obj=model_obj,
            )

        model_endpoint.metadata.uid = exist_model_endpoint.metadata.uid
        attributes = {}
        for attr in mlrun.common.schemas.ModelEndpoint.mutable_fields():
            if attr in [
                "first_request",
                "last_request",
                "feature_names",
                "label_names",
            ]:
                continue
            if model_endpoint.get(attr) != exist_model_endpoint.get(attr):
                attributes[attr] = model_endpoint.get(attr)
        model_endpoint, features = self._enrich_features_from_model_obj(
            db_session=db_session, model_endpoint=model_endpoint, model_obj=model_obj
        )

        if (
            (
                model_endpoint.status.monitoring_mode
                != exist_model_endpoint.status.monitoring_mode
            )
            and model_endpoint.status.monitoring_mode
            == mlrun.common.schemas.ModelMonitoringMode.enabled
            and not model_endpoint.spec.monitoring_feature_set_uri
        ):
            (
                model_endpoint,
                monitoring_feature_set_uri,
            ) = await self._enable_monitoring_on_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                features=features,
            )
            attributes[
                mlrun.common.schemas.ModelEndpointSchema.MONITORING_FEATURE_SET_URI
            ] = monitoring_feature_set_uri
            attributes[mlrun.common.schemas.ModelEndpointSchema.FEATURE_NAMES] = (
                model_endpoint.spec.feature_names
            )
            attributes[mlrun.common.schemas.ModelEndpointSchema.LABEL_NAMES] = (
                model_endpoint.spec.label_names
            )
        elif (
            model_endpoint.status.monitoring_mode
            == exist_model_endpoint.status.monitoring_mode
        ):
            model_endpoint.spec.monitoring_feature_set_uri = (
                exist_model_endpoint.spec.monitoring_feature_set_uri
            )
            model_endpoint.spec.feature_names = exist_model_endpoint.spec.feature_names
            model_endpoint.spec.label_names = exist_model_endpoint.spec.label_names
        if upsert:
            await run_in_threadpool(
                framework.utils.singletons.db.get_db().update_model_endpoint,
                session=db_session,
                project=exist_model_endpoint.metadata.project,
                name=exist_model_endpoint.metadata.name,
                attributes=attributes,
                uid=exist_model_endpoint.metadata.uid,
            )
            return model_endpoint, "", [], {}

        else:
            return model_endpoint, "update", [], attributes

    async def _overwrite_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        model_obj: Optional[mlrun.artifacts.ModelArtifact] = None,
        upsert: bool = True,
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, str, list[str], dict]:
        old_uids = [
            model_endpoint.metadata.uid
            for model_endpoint in (
                await run_in_threadpool(
                    framework.utils.singletons.db.get_db().list_model_endpoints,
                    project=model_endpoint.metadata.project,
                    names=[model_endpoint.metadata.name],
                    function_name=model_endpoint.spec.function_name,
                    function_tag=model_endpoint.spec.function_tag,
                    latest_only=True,
                    session=db_session,
                )
            ).endpoints
        ]

        model_endpoint, method, _, _ = await self._archive_model_endpoint(
            db_session, model_endpoint, model_obj, upsert=upsert
        )
        if old_uids and upsert:
            # delete old versions
            await run_in_threadpool(
                framework.utils.singletons.db.get_db().delete_model_endpoints,
                session=db_session,
                project=model_endpoint.metadata.project,
                uids=old_uids,
            )
            await run_in_threadpool(
                self._delete_model_endpoint_monitoring_infra,
                uids=old_uids,
                project=model_endpoint.metadata.project,
            )
            return model_endpoint, "", [], {}
        else:
            return model_endpoint, method, old_uids, {}

    async def _archive_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        model_obj: Optional[mlrun.artifacts.ModelArtifact] = None,
        delete_old: bool = False,
        upsert: bool = True,
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, str, list[str], dict]:
        uid_to_delete = []
        if delete_old:
            old_uids = [
                model_endpoint.metadata.uid
                for model_endpoint in (
                    await run_in_threadpool(
                        framework.utils.singletons.db.get_db().list_model_endpoints,
                        project=model_endpoint.metadata.project,
                        names=[model_endpoint.metadata.name],
                        function_name=model_endpoint.spec.function_name,
                        function_tag=model_endpoint.spec.function_tag,
                        latest_only=False,
                        session=db_session,
                        order_by="created",
                    )
                ).endpoints
            ]
            if len(old_uids) >= ARCHIVE_LIMITATION:
                uid_to_delete = old_uids[: len(uid_to_delete) - ARCHIVE_LIMITATION + 1]
        logger.info("Expand model endpoint with features, labels and feature_set")
        model_endpoint, features = self._enrich_features_from_model_obj(
            db_session=db_session, model_endpoint=model_endpoint, model_obj=model_obj
        )
        if (
            model_endpoint.status.monitoring_mode
            == mlrun.common.schemas.ModelMonitoringMode.enabled
        ):
            logger.info("Enable monitoring on model endpoint")
            (
                model_endpoint,
                monitoring_feature_set_uri,
            ) = await self._enable_monitoring_on_model_endpoint(
                db_session=db_session,
                model_endpoint=model_endpoint,
                features=features,
            )
            model_endpoint.spec.monitoring_feature_set_uri = monitoring_feature_set_uri
            logger.info("Finish enable monitoring on model endpoint")
        if upsert:
            if uid_to_delete:
                # delete old versions
                await run_in_threadpool(
                    framework.utils.singletons.db.get_db().delete_model_endpoints,
                    session=db_session,
                    project=model_endpoint.metadata.project,
                    uids=uid_to_delete,
                )
                await run_in_threadpool(
                    self._delete_model_endpoint_monitoring_infra,
                    uids=uid_to_delete,
                    project=model_endpoint.metadata.project,
                )
            await self._create_new_model_endpoint(
                db_session=db_session, model_endpoint=model_endpoint
            )
            return model_endpoint, "", [], {}
        else:
            return model_endpoint, "create", uid_to_delete, {}

    @staticmethod
    async def _create_new_model_endpoint(
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> mlrun.common.schemas.ModelEndpoint:
        logger.info(
            "Creating model endpoint",
            endpoint_id=model_endpoint.metadata.name,
            project=model_endpoint.metadata.project,
            function_name=model_endpoint.spec.function_name,
            function_tag=model_endpoint.spec.function_tag,
        )
        return await run_in_threadpool(
            framework.utils.singletons.db.get_db().store_model_endpoint,
            session=db_session,
            model_endpoint=model_endpoint,
        )

    def _enrich_features_from_model_obj(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        model_obj: Optional[mlrun.artifacts.ModelArtifact] = None,
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, list[mlrun.feature_store.Feature]]:
        features = []
        if model_obj:
            if not model_endpoint.spec.label_names and model_obj.spec.outputs:
                model_label_names = [
                    mlrun.feature_store.api.norm_column_name(f.name)
                    for f in model_obj.spec.outputs
                ]
                model_endpoint.spec.label_names = model_label_names

            if not model_endpoint.spec.feature_names:
                features = self._get_features(
                    model=model_obj,
                    run_db=framework.api.utils.get_run_db_instance(db_session),
                    project=model_endpoint.metadata.project,
                )
                model_endpoint.spec.feature_names = [
                    feature.name
                    for feature in features
                    if feature.name not in model_endpoint.spec.label_names
                ]

        return model_endpoint, features

    async def _enable_monitoring_on_model_endpoint(
        self,
        db_session: sqlalchemy.orm.Session,
        model_endpoint: mlrun.common.schemas.ModelEndpoint,
        features: list[mlrun.feature_store.Feature],
    ) -> tuple[mlrun.common.schemas.ModelEndpoint, str]:
        monitoring_feature_set = await self.create_monitoring_feature_set(
            features=features,
            model_endpoint=model_endpoint,
            db_session=db_session,
        )
        # Link model endpoint object to feature set URI
        model_endpoint.spec.monitoring_feature_set_uri = monitoring_feature_set.uri
        # Create model monitoring json files
        self._create_model_monitoring_json_files(model_endpoint=model_endpoint)

        return model_endpoint, monitoring_feature_set.uri

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

    async def patch_model_endpoint(
        self,
        name: str,
        project: str,
        attributes: dict,
        db_session: sqlalchemy.orm.Session,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> str:
        """
        Update a model endpoint record with a given attributes.

        :param name: The name of the model endpoint.
        :param project: The name of the project.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                   of the attributes dictionary should exist in the DB table. More details about the model
                   endpoint available attributes can be found under
                   :py:class:`~mlrun.common.schemas.ModelEndpoint`.
        :param db_session:             A session that manages the current dialog with the database.
        :param function_name: The name of the function.
        :param function_tag: The tag of the function.
        :param endpoint_id: The unique id of the model endpoint.

        :return: The patched `ModelEndpoint` uid.
        """
        if function_name and function_tag is None:
            logger.info("Function tag not provided, setting to 'latest'")
            function_tag = DEFAULT_FUNCTION_TAG
        uid = await run_in_threadpool(
            framework.utils.singletons.db.get_db().update_model_endpoint,
            session=db_session,
            project=project,
            name=name,
            function_name=function_name,
            function_tag=function_tag,  # default to latest (?)
            attributes=attributes,
            uid=endpoint_id,
        )

        logger.info(
            "Model endpoint table updated",
            name=name,
            project=project,
            function_name=function_name,
            function_tag=function_tag,
            endpoint_id=uid,
        )

        return uid

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
            _, name, _, tag, _, _ = mlrun.utils.helpers.parse_artifact_uri(
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
    async def create_monitoring_feature_set(
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
        await run_in_threadpool(feature_set.save)

        return feature_set

    async def delete_model_endpoint(
        self,
        name: str,
        project: str,
        db_session: sqlalchemy.orm.Session,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """
        Delete the record of a given model endpoint based on endpoint id.

        :param name:          The name of the model endpoint.
        :param project:       The name of the project.
        :param db_session:    A session that manages the current dialog with the database
        :param function_name: The name of the function.
        :param function_tag:  The tag of the function.
        :param endpoint_id:   The unique id of the model endpoint.

        """
        if function_name and function_tag is None:
            logger.info("Function tag not provided, setting to 'latest'")
            function_tag = DEFAULT_FUNCTION_TAG
        if endpoint_id == "*":
            model_endpoint_list = await run_in_threadpool(
                framework.utils.singletons.db.get_db().list_model_endpoints,
                project=project,
                names=[name],
                function_name=function_name,
                function_tag=function_tag,
                latest_only=False,
                session=db_session,
            )
            uids = [
                model_endpoint.metadata.uid
                for model_endpoint in model_endpoint_list.endpoints
            ]
        else:
            uids = [endpoint_id]

        await run_in_threadpool(
            framework.utils.singletons.db.get_db().delete_model_endpoint,
            session=db_session,
            project=project,
            name=name,
            function_name=function_name,
            function_tag=function_tag,
            uid=endpoint_id,
        )
        await run_in_threadpool(
            self._delete_model_endpoint_monitoring_infra, uids=uids, project=project
        )
        logger.info(
            "Model endpoint were delete",
            project=project,
            name=name,
            function_name=function_name,
            function_tag=function_tag,
            amount=len(uids),
        )

    def _delete_model_endpoint_monitoring_infra(self, uids: list[str], project: str):
        """
        Delete the monitoring infrastructure of a given model endpoint based on endpoint id.

        :param uids:          The unique id of the model endpoint.
        :param project:       The name of the project.
        """

        # delete jsons
        for uid in uids:
            ModelMonitoringCurrentStatsFile(project=project, endpoint_id=uid).delete()
            ModelMonitoringDriftMeasuresFile(project=project, endpoint_id=uid).delete()
            ModelMonitoringSchedulesFile(project=project, endpoint_id=uid).delete()

        # delete tsdb records - NOT IMPLEMENTED
        try:
            # todo : delete tsdb records/tables for the model endpoint
            # tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            #     project=project,
            #     secret_provider=services.api.crud.secrets.get_project_secret_provider(
            #         project=project
            #     ),
            # )
            logger.info("TSDB resources were not deleted")
        except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
            logger.info(
                "Failed to delete TSDB resources, you may need to delete them manually",
                error=mlrun.errors.err_to_str(e),
            )

        logger.info(
            "Model endpoint monitoring infrastructure were delete",
            project=project,
            amount=len(uids),
        )

    async def get_model_endpoint(
        self,
        name: str,
        project: str,
        db_session: sqlalchemy.orm.Session,
        function_name: Optional[str] = None,
        function_tag: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        tsdb_metrics: bool = True,
        feature_analysis: bool = False,
    ) -> mlrun.common.schemas.ModelEndpoint:
        """Get a single model endpoint object.

        :param name                        The name of the model endpoint
        :param project:                    The name of the project
        :param db_session:                 A session that manages the current dialog with the database.
        :param function_name:              The name of the function
        :param function_tag:               The tag of the function
        :param endpoint_id:                The unique id of the model endpoint.
        :param tsdb_metrics:               When True, the time series metrics will be added to the output
                                           of the resulting.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.

        :return: A `ModelEndpoint` object.
        :raise: `MLRunNotFoundError` if the model endpoint is not found.
        """

        logger.info(
            "Getting model endpoint record from DB",
            name=name,
            project=project,
            function_name=function_name,
            function_tag=function_tag,
            endpoint_id=endpoint_id,
            tsdb_metrics=tsdb_metrics,
            feature_analysis=feature_analysis,
        )

        # Get the model endpoint record
        model_endpoint_object = await run_in_threadpool(
            framework.utils.singletons.db.get_db().get_model_endpoint,
            session=db_session,
            project=project,
            name=name,
            function_name=function_name,
            function_tag=function_tag,
            uid=endpoint_id,
        )

        # If time metrics were provided, retrieve the results from the time series DB
        if tsdb_metrics:
            logger.info("Adding real time metrics to the model endpoint")
            model_endpoint_object = (
                await self._add_basic_metrics(
                    model_endpoint_objects=[model_endpoint_object],
                    project=project,
                )
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

    async def list_model_endpoints(
        self,
        project: str,
        db_session: sqlalchemy.orm.Session,
        names: typing.Optional[list[str]] = None,
        model_name: typing.Optional[str] = None,
        model_tag: typing.Optional[str] = None,
        function_name: typing.Optional[str] = None,
        function_tag: typing.Optional[str] = None,
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
        :param names:               A list of the names of the model endpoints.
        :param model_name:          The name of the model.
        :param function_name:       The name of the function.
        :param function_tag:        The tag of the function.
        :param labels:              A list of labels to filter the model endpoints.
        :param start:               The start time of the model endpoint creation.
        :param end:                 The end time of the model endpoint creation.
        :param top_level:           When True, only top level model endpoints will be returned.
        :param tsdb_metrics:        When True, the time series metrics will be added to the output of the resulting
        :param uids:                A list of unique ids of the model endpoints.
        :param latest_only:         When True, only the latest model endpoint will be returned.
        :return:                    A list of `ModelEndpoint` objects.
        """

        if function_name and function_tag is None:
            logger.info("Function tag not provided, setting to 'latest'")
            function_tag = DEFAULT_FUNCTION_TAG

        logger.info(
            "Listing endpoints",
            names=names,
            project=project,
            model_name=model_name,
            model_tag=model_tag,
            function_name=function_name,
            function_tag=function_tag,
            labels=labels,
            start=start,
            end=end,
            top_level=top_level,
            tsdb_metrics=tsdb_metrics,
            uids=uids,
            latest_only=latest_only,
        )

        # Initialize an empty model endpoints list
        endpoint_list = await run_in_threadpool(
            framework.utils.singletons.db.get_db().list_model_endpoints,
            session=db_session,
            project=project,
            names=names,
            model_name=model_name,
            model_tag=model_tag,
            function_name=function_name,
            function_tag=function_tag,
            labels=labels,
            start=start,
            end=end,
            top_level=top_level,
            uids=uids,
            latest_only=latest_only,
        )

        if tsdb_metrics and endpoint_list.endpoints:
            endpoint_list.endpoints = await self._add_basic_metrics(
                model_endpoint_objects=endpoint_list.endpoints,
                project=project,
            )

        return endpoint_list

    @classmethod
    def delete_model_endpoint_monitoring_resources(
        cls,
        *,
        project_name: str,
        db_session: sqlalchemy.orm.Session,
        stream_profile: mlrun.datastore.datastore_profile.DatastoreProfile,
        tsdb_profile: mlrun.datastore.datastore_profile.DatastoreProfile,
        model_monitoring_applications: typing.Optional[list[str]] = None,
        model_monitoring_access_key: typing.Optional[str] = None,
    ) -> None:
        """
        Delete all model endpoints monitoring resources, including the store data, time series data, and stream
        resources.
        This function is called only when the caller knows there is model monitoring for the project.

        :param project_name:                  The name of the project.
        :param db_session:                    A session that manages the current dialog with the database.
        :param stream_profile:                The datastore profile for the stream.
        :param tsdb_profile:                  The datastore profile for the TSDB.
        :param model_monitoring_applications: A list of model monitoring applications that their resources should
                                              be deleted.
        :param model_monitoring_access_key:   The access key for the model monitoring resources. Relevant only for
                                              V3IO resources.
        """
        logger.debug(
            "Deleting model monitoring endpoints resources", project_name=project_name
        )
        stream_path = mlrun.model_monitoring.get_stream_path(
            project=project_name, profile=stream_profile
        )

        # We would ideally base on config.v3io_api but can't for backwards compatibility reasons,
        # we're using the igz version heuristic
        # TODO : adjust for ce scenario
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
                project=project_name, profile=tsdb_profile
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
                    profile=mlrun.datastore.datastore_profile.DatastoreProfileV3io(
                        name="tmp"
                    ),
                )
            else:
                tsdb_connector = None
        if tsdb_connector:
            tsdb_connector.delete_tsdb_resources()
        cls._delete_model_monitoring_stream_resources(
            project_name=project_name,
            model_monitoring_applications=model_monitoring_applications,
            stream_profile=stream_profile,
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
        endpoint_id: typing.Union[str, list[str]],
        type: str,
        metrics_format: str = mm_constants.GetEventsFormat.SINGLE,
    ) -> typing.Union[
        list[mm_endpoints.ModelEndpointMonitoringMetric],
        dict[str, list[mm_endpoints.ModelEndpointMonitoringMetric]],
    ]:
        """
        Get the metrics for a given model endpoint.

        :param project:         The name of the project.
        :param endpoint_id:     The unique id of the model endpoint, Can be a single id or a list of ids.
        :param type:            metric or result.
        :param metrics_format:  Determines the format of the result. Can be either 'list' or 'dict'.
        :return: metrics in the chosen format.
        """
        try:
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=project,
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
                    project=project
                ),
            )
        except mlrun.errors.MLRunNotFoundError as e:
            logger.debug(
                f"Failed to list model endpoint {type}s because TSDB profile was not found. "
                "Returning an empty list of metrics",
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

        if metrics_format == mm_constants.GetEventsFormat.SINGLE:
            return tsdb_connector.df_to_metrics_list(df=df, type=type, project=project)
        elif metrics_format == mm_constants.GetEventsFormat.SEPARATION:
            return tsdb_connector.df_to_metrics_grouped_dict(
                df=df, type=type, project=project
            )
        elif metrics_format == mm_constants.GetEventsFormat.INTERSECTION:
            return tsdb_connector.df_to_events_intersection_dict(
                df=df, type=type, project=project
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid metrics_format. It must be one of: {', '.join(mm_constants.GetEventsFormat)}."
            )

    @staticmethod
    def _delete_model_monitoring_stream_resources(
        project_name: str,
        model_monitoring_applications: typing.Optional[list[str]],
        stream_profile: mlrun.datastore.datastore_profile.DatastoreProfile,
    ) -> None:
        """
        Delete model monitoring stream resources.

        :param project_name:                  The name of the project.
        :param model_monitoring_applications: A list of model monitoring applications that their resources should
                                              be deleted.
        :param stream_profile:                The datastore profile for the stream.
        """
        logger.debug(
            "Deleting model monitoring stream resources",
            project_name=project_name,
        )

        model_monitoring_applications = model_monitoring_applications or []

        # Add the writer, controller, and monitoring stream to the application streams list
        model_monitoring_applications.extend(
            mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.list()
        )

        try:
            services.api.crud.model_monitoring.deployment.MonitoringDeployment(
                project=project_name
            )._delete_model_monitoring_stream_resources(
                function_names=model_monitoring_applications,
                stream_profile=stream_profile,
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
        except mlrun.errors.MLRunNotFoundError as e:
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
            if (
                mep.status.monitoring_mode
                == mlrun.common.schemas.ModelMonitoringMode.enabled
            ):
                mep.status.current_stats, mep.status.current_stats_timestamp = (
                    ModelMonitoringCurrentStatsFile.from_model_endpoint(mep).read()
                )

                mep.status.drift_measures, mep.status.drift_measures_timestamp = (
                    ModelMonitoringDriftMeasuresFile.from_model_endpoint(mep).read()
                )
        return model_endpoint_objects

    async def _add_basic_metrics(
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

        try:
            tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
                project=project,
                secret_provider=services.api.crud.secrets.get_project_secret_provider(
                    project=project
                ),
            )
        except mlrun.errors.MLRunNotFoundError as e:
            logger.debug(
                "Failed to add basic metrics because the TSDB profile was not found. "
                "Returning without adding the basic metrics.",
                error=mlrun.errors.err_to_str(e),
            )
            return model_endpoint_objects

        return await tsdb_connector.add_basic_metrics(
            model_endpoint_objects, project, run_in_threadpool
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


class ModelMonitoringResourcesDeleter:
    def __init__(
        self,
        *,
        project: str,
        db_session: typing.Optional[sqlalchemy.orm.Session],
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo],
        model_monitoring_access_key: typing.Optional[str],
    ) -> None:
        self._project = project
        self._db_session = db_session
        self._auth_info = auth_info
        self._model_monitoring_access_key = model_monitoring_access_key

        # get model monitoring application names, important for deleting model monitoring resources
        logger.debug("Getting monitoring applications to delete", project_name=project)
        self._model_monitoring_applications = (
            services.api.crud.model_monitoring.deployment.MonitoringDeployment(
                project=project,
                db_session=db_session,
                auth_info=auth_info,
                model_monitoring_access_key=model_monitoring_access_key,
            )
        )._get_monitoring_application_to_delete(delete_user_applications=True)

        self._secret_provider = services.api.crud.secrets.get_project_secret_provider(
            project=project
        )

        self._has_mm = self._does_project_have_mm()
        self._stream_profile = self._get_stream_profile()
        self._tsdb_profile = self._get_tsdb_profile()

    def _does_project_have_mm(self) -> bool:
        mandatory_secrets = mm_constants.ProjectSecretKeys.mandatory_secrets()
        keys_not_none = [
            self._secret_provider(key) is not None for key in mandatory_secrets
        ]
        has_mm = all(keys_not_none)

        if not has_mm and any(keys_not_none):
            # An unexpected situation in which only some of the mandatory MM secrets are set
            set_secrets = [
                secret
                for (not_none, secret) in zip(keys_not_none, mandatory_secrets)
                if not_none
            ]
            logger.warn(
                "Not all of the mandatory model monitoring secrets are set in the project's secrets. "
                "Assuming the project has no model monitoring in place.",
                project_name=self._project,
                mandatory_secrets=mandatory_secrets,
                set_secrets=set_secrets,
            )

        return has_mm

    def _get_profile(
        self, get_profile_function: Callable
    ) -> Optional[mlrun.datastore.datastore_profile.DatastoreProfile]:
        if not self._has_mm:
            return
        try:
            return get_profile_function(
                project=self._project, secret_provider=self._secret_provider
            )
        except mlrun.errors.MLRunNotFoundError as err:
            # An unexpected situation in which the secrets are set but the datastore profile was removed
            logger.warn(
                "The project is marked as having model monitoring according to the secrets, "
                "but the profile was not found.",
                project_name=self._project,
                err=mlrun.errors.err_to_str(err),
            )

    def _get_stream_profile(
        self,
    ) -> Optional[mlrun.datastore.datastore_profile.DatastoreProfile]:
        return self._get_profile(
            get_profile_function=mlrun.model_monitoring.helpers._get_stream_profile
        )

    def _get_tsdb_profile(
        self,
    ) -> Optional[mlrun.datastore.datastore_profile.DatastoreProfile]:
        return self._get_profile(
            get_profile_function=mlrun.model_monitoring.helpers._get_tsdb_profile
        )

    def delete(self) -> None:
        if not self._stream_profile or not self._tsdb_profile:
            logger.debug(
                "No model monitoring resources were found in this project",
                project_name=self._project,
            )
            return
        try:
            # Delete model monitoring resources
            ModelEndpoints.delete_model_endpoint_monitoring_resources(
                project_name=self._project,
                db_session=self._db_session,
                tsdb_profile=self._tsdb_profile,
                stream_profile=self._stream_profile,
                model_monitoring_applications=self._model_monitoring_applications,
                model_monitoring_access_key=self._model_monitoring_access_key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete model monitoring resources",
                project_name=self._project,
            )
            raise exc
