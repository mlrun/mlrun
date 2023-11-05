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
import asyncio
import copy
from typing import Dict

import fastapi.concurrency

import mlrun.runtimes.constants
import server.api.crud.runs
import server.api.db.session
from mlrun import mlconf
from mlrun.runtimes.utils import mlrun_key


def get_resource_labels(function, run=None, scrape_metrics=None):
    scrape_metrics = (
        scrape_metrics if scrape_metrics is not None else mlconf.scrape_metrics
    )
    run_uid, run_name, run_project, run_owner = None, None, None, None
    if run:
        run_uid = run.metadata.uid
        run_name = run.metadata.name
        run_project = run.metadata.project
        run_owner = run.metadata.labels.get("owner")
    labels = copy.deepcopy(function.metadata.labels)
    labels[mlrun_key + "class"] = function.kind
    labels[mlrun_key + "project"] = run_project or function.metadata.project
    labels[mlrun_key + "function"] = str(function.metadata.name)
    labels[mlrun_key + "tag"] = str(function.metadata.tag or "latest")
    labels[mlrun_key + "scrape-metrics"] = str(scrape_metrics)

    if run_uid:
        labels[mlrun_key + "uid"] = run_uid

    if run_name:
        labels[mlrun_key + "name"] = run_name

    if run_owner:
        labels[mlrun_key + "owner"] = run_owner

    return labels


def abort_run(
    run: Dict,
    uid: str = None,
    project: str = None,
    iter: int = 0,
    status_text: str = None,
):
    uid = uid or run["metadata"]["uid"]
    project = project or run["metadata"]["project"]

    run.setdefault("status", {})
    run["status"]["state"] = mlrun.runtimes.constants.RunStates.aborted
    if status_text:
        run["status"]["status_text"] = status_text

    return asyncio.create_task(
        fastapi.concurrency.run_in_threadpool(
            server.api.db.session.run_function_with_new_db_session,
            server.api.crud.runs.Runs().update_run,
            project=project,
            uid=uid,
            iter=iter,
            data=run,
        )
    )
