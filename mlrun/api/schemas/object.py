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
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Extra

import mlrun.api.utils.helpers


class ObjectMetadata(BaseModel):
    name: str
    project: Optional[str]
    tag: Optional[str]
    labels: Optional[dict]
    updated: Optional[datetime]
    created: Optional[datetime]
    uid: Optional[str]

    class Config:
        extra = Extra.allow


class ObjectStatus(BaseModel):
    state: Optional[str]

    class Config:
        extra = Extra.allow


class ObjectSpec(BaseModel):
    class Config:
        extra = Extra.allow


class LabelRecord(BaseModel):
    id: int
    name: str
    value: str

    class Config:
        orm_mode = True


class ObjectRecord(BaseModel):
    id: int
    name: str
    project: str
    uid: str
    updated: Optional[datetime] = None
    labels: List[LabelRecord]
    # state is extracted from the full status dict to enable queries
    state: Optional[str] = None
    full_object: Optional[dict] = None

    class Config:
        orm_mode = True


class ObjectKind(mlrun.api.utils.helpers.StrEnum):
    project = "project"
    feature_set = "FeatureSet"
    background_task = "BackgroundTask"
    feature_vector = "FeatureVector"
    model_endpoint = "model-endpoint"
    marketplace_source = "MarketplaceSource"
    marketplace_item = "MarketplaceItem"
    marketplace_catalog = "MarketplaceCatalog"
