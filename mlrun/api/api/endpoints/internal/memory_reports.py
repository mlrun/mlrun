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
import fastapi

import mlrun.api.schemas
import mlrun.api.utils.memory_reports

router = fastapi.APIRouter()


@router.get(
    "/memory-reports/common-types",
    response_model=mlrun.api.schemas.MostCommonObjectTypesReport,
)
def get_most_common_objects_report():
    report = (
        mlrun.api.utils.memory_reports.MemoryUsageReport().create_most_common_objects_report()
    )
    return mlrun.api.schemas.MostCommonObjectTypesReport(object_types=report)


@router.get(
    "/memory-reports/{object_type}",
    response_model=mlrun.api.schemas.ObjectTypeReport,
)
def get_memory_usage_report(
    object_type: str,
    sample_size: int = 1,
    start_index: int = None,
    create_graph: bool = False,
    max_depth: int = 3,
):
    report = (
        mlrun.api.utils.memory_reports.MemoryUsageReport().create_memory_usage_report(
            object_type, sample_size, start_index, create_graph, max_depth
        )
    )
    return mlrun.api.schemas.ObjectTypeReport(
        object_type=object_type,
        sample_size=sample_size,
        start_index=start_index,
        max_depth=max_depth,
        object_report=report,
    )
