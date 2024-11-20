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
from datetime import datetime, timezone
from typing import Optional

from pydantic.v1 import BaseModel, Extra, Field

import mlrun.common.types
import mlrun.config
import mlrun.errors
from mlrun.common.schemas.object import ObjectKind, ObjectSpec, ObjectStatus


# Defining a different base class (not ObjectMetadata), as there's no project, and it differs enough to
# justify a new class
class HubObjectMetadata(BaseModel):
    name: str
    description: str = ""
    labels: Optional[dict] = {}
    updated: Optional[datetime]
    created: Optional[datetime]

    class Config:
        extra = Extra.allow


# Currently only functions are supported. Will add more in the future.
class HubSourceType(mlrun.common.types.StrEnum):
    functions = "functions"


# Sources-related objects
class HubSourceSpec(ObjectSpec):
    path: str  # URL to base directory, should include schema (s3://, etc...)
    channel: str
    credentials: Optional[dict] = {}
    object_type: HubSourceType = Field(HubSourceType.functions, const=True)


class HubSource(BaseModel):
    kind: ObjectKind = Field(ObjectKind.hub_source, const=True)
    metadata: HubObjectMetadata
    spec: HubSourceSpec
    status: Optional[ObjectStatus] = ObjectStatus(state="created")

    def get_full_uri(self, relative_path):
        return f"{self.spec.path}/{self.spec.object_type}/{self.spec.channel}/{relative_path}"

    def get_catalog_uri(self):
        return self.get_full_uri(mlrun.mlconf.hub.catalog_filename)

    @classmethod
    def generate_default_source(cls):
        if not mlrun.mlconf.hub.default_source.create:
            return None

        now = datetime.now(timezone.utc)
        hub_metadata = HubObjectMetadata(
            name=mlrun.mlconf.hub.default_source.name,
            description=mlrun.mlconf.hub.default_source.description,
            created=now,
            updated=now,
        )
        return cls(
            metadata=hub_metadata,
            spec=HubSourceSpec(
                path=mlrun.mlconf.hub.default_source.url,
                channel=mlrun.mlconf.hub.default_source.channel,
                object_type=HubSourceType(mlrun.mlconf.hub.default_source.object_type),
            ),
            status=ObjectStatus(state="created"),
        )


last_source_index = -1


class IndexedHubSource(BaseModel):
    index: int = last_source_index  # Default last. Otherwise, must be > 0
    source: HubSource


# Item-related objects
class HubItemMetadata(HubObjectMetadata):
    source: HubSourceType = Field(HubSourceType.functions, const=True)
    version: str
    tag: Optional[str]

    def get_relative_path(self) -> str:
        if self.source == HubSourceType.functions:
            # This is needed since the hub deployment script modifies the paths to use _ instead of -.
            modified_name = self.name.replace("-", "_")
            # Prefer using the tag if exists. Otherwise, use version.
            version = self.tag or self.version
            return f"{modified_name}/{version}/"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Bad source for hub item - {self.source}"
            )


class HubItemSpec(ObjectSpec):
    item_uri: str
    assets: dict[str, str] = {}


class HubItem(BaseModel):
    kind: ObjectKind = Field(ObjectKind.hub_item, const=True)
    metadata: HubItemMetadata
    spec: HubItemSpec
    status: ObjectStatus


class HubCatalog(BaseModel):
    kind: ObjectKind = Field(ObjectKind.hub_catalog, const=True)
    channel: str
    catalog: list[HubItem]
