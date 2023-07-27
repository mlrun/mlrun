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
import asyncio
import typing

import mlrun
import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class MLRunJobNotification(NotificationBase):
    """
    API/Client notification for running a new mlrun job
    """

    async def push(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        run_template = self.params.get("run_template", {})
        auth_info = None
        if "auth_info" in self.params:
            auth_info = mlrun.common.schemas.AuthInfo(
                username=self.params["auth_info"].get("username", None),
                access_key=self.params["auth_info"].get("access_key", None),
            )

        # enrich run template with project from run if not already set
        run_template.setdefault("metadata", {}).setdefault(
            "project", runs[0]["metadata"]["project"]
        )

        run_db = mlrun.get_run_db()
        if asyncio.iscoroutine(run_db.submit_job):
            # async submit job on api side
            await run_db.submit_job(
                run_template,
                auth_info=auth_info,
            )
        else:
            # sync submit job on client side which still uses sync api requests
            run_db.submit_job(
                run_template,
                auth_info=auth_info,
            )
