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
import typing

import pydantic

from .artifact import ArtifactObject


class TagObject(pydantic.BaseModel):
    """Tag object"""

    kind: str
    identifiers: typing.List[typing.Union[ArtifactObject]]


class TagsObjects(pydantic.BaseModel):
    """
    Tags objects list
    """

    objects: typing.List[TagObject]


"""
{
  "objects": [
        {
          "kind": artifact,
          "identifiers": [
                {
                  "id": "123",
                },
                {
                  "path": "path",
                  "tag": "tag",
                },
                {
                  "key": "key",
                }
           ],
        }
    ]
}
"""
