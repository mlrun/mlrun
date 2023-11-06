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
from typing import Dict

import fastapi.concurrency

import mlrun.runtimes.constants
import server.api.crud.runs
import server.api.db.session


def abort_run(
    run: Dict,
    uid: str = None,
    project: str = None,
    iter: int = 0,
    status_text: str = None,
) -> asyncio.Task:
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
