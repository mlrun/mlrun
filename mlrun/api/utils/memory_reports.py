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

import gc
import io
import typing

import objgraph

import mlrun.config
from mlrun.api.utils.periodic import cancel_periodic_function, run_function_periodically
from mlrun.utils import logger

PERIODIC_REPORT_FUNCTION_NAME = "periodic_memory_reports"


def start_periodic_memory_reports():
    def _log_report():
        logger.debug(
            "Memory Report",
            most_common_objects=create_most_common_objects_report(),
        )

    interval = int(mlrun.config.config.memory_report_interval)
    if interval > 0:
        logger.info("Starting periodic memory reports loop", interval=interval)
        run_function_periodically(
            interval,
            PERIODIC_REPORT_FUNCTION_NAME,
            False,
            _log_report,
        )


def stop_periodic_memory_reports():
    logger.info("Stopping periodic memory reports loop")
    cancel_periodic_function(PERIODIC_REPORT_FUNCTION_NAME)


def create_most_common_objects_report() -> typing.List[typing.Tuple[str, int]]:
    gc.collect()
    return objgraph.most_common_types()


def create_memory_usage_report(
    object_type: str,
    sample_size: int = 1,
    start_index: int = None,
    create_graph: bool = False,
    max_depth: int = 3,
) -> typing.List[typing.Dict[str, typing.Any]]:

    gc.collect()
    report = []
    requested_objects = objgraph.by_type(object_type)

    # If 'start_index' not given use 'sample_size' to calculated it from the end of the list
    if start_index is None or start_index < 0:
        start_index = len(requested_objects) - sample_size

    if start_index < len(requested_objects):
        # Iterate until 'sample_size' or the end of the list is reached
        for object_index in range(
            start_index,
            min(start_index + sample_size, len(requested_objects)),
        ):
            report.append(
                _create_object_report(
                    requested_objects,
                    object_index,
                    object_type,
                    max_depth,
                    create_graph,
                )
            )
    else:
        logger.warn(
            "Object start index is invalid",
            object_type=object_type,
            start_index=start_index,
            total_objects=len(requested_objects),
        )

    return report


def _create_object_report(
    requested_objects: typing.List[dict],
    object_index: int,
    object_type: str,
    max_depth: int = 3,
    create_graph: bool = False,
) -> typing.Dict[str, typing.Any]:
    object_report = {
        "object": str(requested_objects[object_index])[:10000],
        "total_objects": len(requested_objects),
        "object_index": object_index,
        "object_type": object_type,
    }
    logger.debug(
        "Object report",
        **object_report,
    )

    if create_graph:
        logger.debug(
            "Creating reference graph",
            object_index=object_index,
            max_depth=max_depth,
        )
        graph = _create_object_ref_graph(
            requested_objects[object_index],
            max_depth=max_depth,
        )
        object_report["graph"] = graph

    return object_report


def _create_object_ref_graph(obj: object, max_depth: int = 3) -> str:
    graph_output = io.StringIO()
    objgraph.show_backrefs(
        obj, refcounts=True, max_depth=max_depth, output=graph_output
    )

    return graph_output.getvalue()
