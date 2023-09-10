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

import sqlalchemy.dialects.mysql
from sqlalchemy import Column
from sqlalchemy.ext.declarative import declarative_base

from mlrun.common.schemas.model_monitoring import EventFieldType

from .base import ModelEndpointsBaseTable

Base = declarative_base()


class ModelEndpointsTable(Base, ModelEndpointsBaseTable):
    first_request = Column(
        EventFieldType.FIRST_REQUEST,
        sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
    last_request = Column(
        EventFieldType.LAST_REQUEST,
        sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3),
    )
