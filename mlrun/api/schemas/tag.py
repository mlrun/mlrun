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
