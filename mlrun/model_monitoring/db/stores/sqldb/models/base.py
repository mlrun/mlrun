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

from sqlalchemy import (
    TIMESTAMP,  # TODO: migrate to DATETIME, see ML-6921
    Boolean,
    Column,
    Integer,
    String,
    Text,
)

from mlrun.common.schemas.model_monitoring import (
    EventFieldType,
)
from mlrun.utils.db import BaseModel


class ModelEndpointsBaseTable(BaseModel):
    __tablename__ = EventFieldType.MODEL_ENDPOINTS

    uid = Column(
        EventFieldType.UID,
        String(40),
        primary_key=True,
    )
    state = Column(EventFieldType.STATE, String(10))
    project = Column(EventFieldType.PROJECT, String(40))
    function_uri = Column(
        EventFieldType.FUNCTION_URI,
        String(255),
    )
    model = Column(EventFieldType.MODEL, String(255))
    model_class = Column(
        EventFieldType.MODEL_CLASS,
        String(255),
    )
    labels = Column(EventFieldType.LABELS, Text)
    model_uri = Column(EventFieldType.MODEL_URI, String(255))
    stream_path = Column(EventFieldType.STREAM_PATH, Text)
    algorithm = Column(
        EventFieldType.ALGORITHM,
        String(255),
    )
    active = Column(EventFieldType.ACTIVE, Boolean)
    monitoring_mode = Column(
        EventFieldType.MONITORING_MODE,
        String(10),
    )
    feature_stats = Column(EventFieldType.FEATURE_STATS, Text)
    current_stats = Column(EventFieldType.CURRENT_STATS, Text)
    feature_names = Column(EventFieldType.FEATURE_NAMES, Text)
    children = Column(EventFieldType.CHILDREN, Text)
    label_names = Column(EventFieldType.LABEL_NAMES, Text)
    endpoint_type = Column(
        EventFieldType.ENDPOINT_TYPE,
        String(10),
    )
    children_uids = Column(EventFieldType.CHILDREN_UIDS, Text)
    drift_measures = Column(EventFieldType.DRIFT_MEASURES, Text)
    drift_status = Column(
        EventFieldType.DRIFT_STATUS,
        String(40),
    )
    monitor_configuration = Column(
        EventFieldType.MONITOR_CONFIGURATION,
        Text,
    )
    monitoring_feature_set_uri = Column(
        EventFieldType.FEATURE_SET_URI,
        String(255),
    )
    error_count = Column(EventFieldType.ERROR_COUNT, Integer)
    metrics = Column(EventFieldType.METRICS, Text)
    first_request = Column(
        EventFieldType.FIRST_REQUEST,
        TIMESTAMP(timezone=True),  # TODO: migrate to DATETIME, see ML-6921
    )
    last_request = Column(
        EventFieldType.LAST_REQUEST,
        TIMESTAMP(timezone=True),  # TODO: migrate to DATETIME, see ML-6921
    )
