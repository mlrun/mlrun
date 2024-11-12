# Copyright 2024 Iguazio
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

import mlrun.common.types
import mlrun_pipelines.common.ops
import mlrun_pipelines.models

from .base import ObjectFormat


class PipelineFormat(ObjectFormat, mlrun.common.types.StrEnum):
    full = "full"
    metadata_only = "metadata_only"
    name_only = "name_only"
    summary = "summary"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        def _full(run: mlrun_pipelines.models.PipelineRun) -> dict:
            return run.to_dict()

        def _metadata_only(run: mlrun_pipelines.models.PipelineRun) -> dict:
            return mlrun.utils.helpers.format_run(run, with_project=True)

        def _name_only(run: mlrun_pipelines.models.PipelineRun) -> str:
            return run.get("name")

        def _summary(run: mlrun_pipelines.models.PipelineRun) -> dict:
            return mlrun_pipelines.common.ops.format_summary_from_kfp_run(
                run, run["project"]
            )

        return {
            PipelineFormat.full: _full,
            PipelineFormat.metadata_only: _metadata_only,
            PipelineFormat.name_only: _name_only,
            PipelineFormat.summary: _summary,
        }[_format]
