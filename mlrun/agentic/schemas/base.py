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

from datetime import datetime
from enum import Enum
from http.client import HTTPException
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel

metadata_fields = [
    "uid",
    "name",
    "description",
    "labels",
    "owner_id",
    "created",
    "updated",
    "version",
    "project_id",
]


class Base(BaseModel):
    _extra_fields = []
    _top_level_fields = []

    def to_dict(
        self, drop_none=True, short=False, drop_metadata=False, to_datestr=False
    ):
        struct = self.model_dump(mode="json")
        new_struct = {}
        for k, v in struct.items():
            if (
                (drop_none and v is None)
                or (short and k in self._extra_fields)
                or (drop_metadata and k in metadata_fields)
            ):
                continue
            if to_datestr and isinstance(v, datetime):
                v = v.isoformat()
            elif short and isinstance(v, datetime):
                v = v.strftime("%Y-%m-%d %H:%M")
            if hasattr(v, "to_dict"):
                v = v.to_dict()
            new_struct[k] = v
        return new_struct

    @classmethod
    def from_dict(cls, data: dict):
        if isinstance(data, cls):
            return data
        return cls.model_validate(data)

    def to_yaml(self, drop_none=True):
        return yaml.dump(self.to_dict(drop_none=drop_none))

    def __repr__(self):
        args = ", ".join(
            [
                f"{k}={v!r}"
                for k, v in self.to_dict(short=True, to_datestr=True).items()
            ]
        )
        return f"{self.__class__.__name__}({args})"

    def __str__(self):
        return str(self.to_dict(to_datestr=True))


class BaseWithMetadata(Base):
    name: str
    uid: Optional[str] = None
    description: Optional[str] = None
    labels: Optional[Dict[str, Union[str, None]]] = None
    created: Optional[Union[str, datetime]] = None
    updated: Optional[Union[str, datetime]] = None


class BaseWithOwner(BaseWithMetadata):
    owner_id: str


class BaseWithVerMetadata(BaseWithOwner):
    version: str = ""


class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def with_raise(self, format=None) -> "APIResponse":
        if not self.success:
            format = format or "API call failed: %s"
            raise ValueError(format % self.error)
        return self

    def with_raise_http(self, format=None) -> "APIResponse":
        if not self.success:
            format = format or "API call failed: %s"
            raise HTTPException(status_code=400, detail=format % self.error)
        return self


class APIDictResponse(APIResponse):
    data: Optional[dict] = None


class OutputMode(str, Enum):
    NAMES = "names"
    SHORT = "short"
    DICT = "dict"
    DETAILS = "details"
