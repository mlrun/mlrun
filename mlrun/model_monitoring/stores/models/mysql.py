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


import sqlalchemy.dialects
from sqlalchemy import Boolean, Column, Integer, String, Text

import mlrun.common.schemas.model_monitoring
from mlrun.utils.db import BaseModel

from .base import Base


class ModelEndpointsTable(Base, BaseModel):
    __tablename__ = mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_ENDPOINTS

    uid = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.UID,
        String(40),
        primary_key=True,
    )
    state = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.STATE, String(10)
    )
    project = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.PROJECT, String(40)
    )
    function_uri = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.FUNCTION_URI,
        String(255),
    )
    model = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.MODEL, String(255)
    )
    model_class = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_CLASS,
        String(255),
    )
    labels = Column(mlrun.common.schemas.model_monitoring.EventFieldType.LABELS, Text)
    model_uri = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_URI, String(255)
    )
    stream_path = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.STREAM_PATH, Text
    )
    algorithm = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.ALGORITHM,
        String(255),
    )
    active = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.ACTIVE, Boolean
    )
    monitoring_mode = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.MONITORING_MODE,
        String(10),
    )
    feature_stats = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_STATS, Text
    )
    current_stats = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.CURRENT_STATS, Text
    )
    feature_names = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_NAMES, Text
    )
    children = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.CHILDREN, Text
    )
    label_names = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.LABEL_NAMES, Text
    )

    endpoint_type = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_TYPE,
        String(10),
    )
    children_uids = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.CHILDREN_UIDS, Text
    )
    drift_measures = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.DRIFT_MEASURES, Text
    )
    drift_status = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.DRIFT_STATUS,
        String(40),
    )
    monitor_configuration = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.MONITOR_CONFIGURATION,
        Text,
    )
    monitoring_feature_set_uri = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_SET_URI,
        String(255),
    )
    first_request = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.FIRST_REQUEST,
        sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    last_request = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.LAST_REQUEST,
        sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    error_count = Column(
        mlrun.common.schemas.model_monitoring.EventFieldType.ERROR_COUNT, Integer
    )
    metrics = Column(mlrun.common.schemas.model_monitoring.EventFieldType.METRICS, Text)
