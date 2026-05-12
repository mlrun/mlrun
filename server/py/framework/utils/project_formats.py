# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Server-side project format types for custom column selection.
Not exposed in mlrun.common so the client/SDK does not depend on them.
"""

import typing

import mlrun.common.formatters
import mlrun.common.schemas


class ProjectFormatCustom:
    """
    Selectable project columns for custom format.
    Values match Project DB column names so callers can build a custom selection
    and add only selected fields to the DB query.

    Adding columns here requiring mapping them below under ProjectFormatCustomSelection.build()
    """

    name = "name"
    created = "created"
    owner = "owner"
    state = "state"
    op_id = "op_id"
    phase = "phase"
    updated_at = "updated_at"

    all_columns: typing.ClassVar[tuple[str, ...]] = (
        "name",
        "created",
        "owner",
        "state",
        "op_id",
        "phase",
        "updated_at",
    )


class ProjectFormatCustomSelection:
    """
    Custom project format with a specific set of columns.
    Pass an instance as format_ to get_project / list_projects to request only
    those columns. Downstream code uses isinstance() and format_.columns to build
    the DB query and response.
    """

    __slots__ = ("columns",)

    def __init__(self, columns: list[str]):
        columns = list(columns)
        invalid = [c for c in columns if c not in ProjectFormatCustom.all_columns]
        if invalid:
            raise ValueError(
                f"Invalid custom project columns: {invalid}. "
                f"Allowed: {list(ProjectFormatCustom.all_columns)}"
            )
        self.columns = columns

    def __contains__(self, column: str) -> bool:
        return column in self.columns

    def build(self, project_dict: dict) -> mlrun.common.schemas.Project:
        return mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(
                name=project_dict.get("name"),
                created=project_dict.get("created"),
            ),
            spec=mlrun.common.schemas.ProjectSpec(
                owner=project_dict.get("owner"),
            ),
            status=mlrun.common.schemas.ProjectStatus(
                state=project_dict.get("state"),
                op_id=project_dict.get("op_id"),
                phase=project_dict.get("phase"),
                updated_at=project_dict.get("updated_at"),
            ),
        )


# Type for format_ parameter: either a named format or a custom column selection.
ProjectFormatType = typing.Union[
    mlrun.common.formatters.ProjectFormat,
    ProjectFormatCustomSelection,
]
