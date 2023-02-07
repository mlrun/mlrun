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


from sqlalchemy import TIMESTAMP, Boolean, Column, Integer, String, Text

import mlrun.model_monitoring.constants as model_monitoring_constants
from mlrun.utils.db import BaseModel

from .base import Base


class ModelEndpointsTable(Base, BaseModel):
    __tablename__ = model_monitoring_constants.EventFieldType.MODEL_ENDPOINTS

    uid = Column(
        model_monitoring_constants.EventFieldType.UID,
        String(40),
        primary_key=True,
    )
    state = Column(model_monitoring_constants.EventFieldType.STATE, String(10))
    project = Column(model_monitoring_constants.EventFieldType.PROJECT, String(40))
    function_uri = Column(
        model_monitoring_constants.EventFieldType.FUNCTION_URI,
        String(255),
    )
    model = Column(model_monitoring_constants.EventFieldType.MODEL, String(255))
    model_class = Column(
        model_monitoring_constants.EventFieldType.MODEL_CLASS,
        String(255),
    )
    labels = Column(model_monitoring_constants.EventFieldType.LABELS, Text)
    model_uri = Column(model_monitoring_constants.EventFieldType.MODEL_URI, String(255))
    stream_path = Column(model_monitoring_constants.EventFieldType.STREAM_PATH, Text)
    algorithm = Column(
        model_monitoring_constants.EventFieldType.ALGORITHM,
        String(255),
    )
    active = Column(model_monitoring_constants.EventFieldType.ACTIVE, Boolean)
    monitoring_mode = Column(
        model_monitoring_constants.EventFieldType.MONITORING_MODE,
        String(10),
    )
    feature_stats = Column(
        model_monitoring_constants.EventFieldType.FEATURE_STATS, Text
    )
    current_stats = Column(
        model_monitoring_constants.EventFieldType.CURRENT_STATS, Text
    )
    feature_names = Column(
        model_monitoring_constants.EventFieldType.FEATURE_NAMES, Text
    )
    children = Column(model_monitoring_constants.EventFieldType.CHILDREN, Text)
    label_names = Column(model_monitoring_constants.EventFieldType.LABEL_NAMES, Text)
    endpoint_type = Column(
        model_monitoring_constants.EventFieldType.ENDPOINT_TYPE,
        String(10),
    )
    children_uids = Column(
        model_monitoring_constants.EventFieldType.CHILDREN_UIDS, Text
    )
    drift_measures = Column(
        model_monitoring_constants.EventFieldType.DRIFT_MEASURES, Text
    )
    drift_status = Column(
        model_monitoring_constants.EventFieldType.DRIFT_STATUS,
        String(40),
    )
    monitor_configuration = Column(
        model_monitoring_constants.EventFieldType.MONITOR_CONFIGURATION,
        Text,
    )
    monitoring_feature_set_uri = Column(
        model_monitoring_constants.EventFieldType.FEATURE_SET_URI,
        String(255),
    )
    first_request = Column(
        model_monitoring_constants.EventFieldType.FIRST_REQUEST,
        TIMESTAMP,
    )
    last_request = Column(
        model_monitoring_constants.EventFieldType.LAST_REQUEST,
        TIMESTAMP,
    )
    error_count = Column(model_monitoring_constants.EventFieldType.ERROR_COUNT, Integer)
    metrics = Column(model_monitoring_constants.EventFieldType.METRICS, Text)
