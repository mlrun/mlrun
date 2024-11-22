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

import json
import os
import typing

import aiohttp

import mlrun.common.schemas
import mlrun.errors
import mlrun.lists

from .base import NotificationBase


class GitNotification(NotificationBase):
    """
    API/Client notification for setting a rich run statuses git issue comment (github/gitlab)
    """

    @classmethod
    def validate_params(cls, params):
        git_repo = params.get("repo", None)
        git_issue = params.get("issue", None)
        git_merge_request = params.get("merge_request", None)
        token = (
            params.get("token", None)
            or params.get("GIT_TOKEN", None)
            or params.get("GITHUB_TOKEN", None)
        )
        if not git_repo:
            raise ValueError("Parameter 'repo' is required for GitNotification")

        if not token:
            raise ValueError("Parameter 'token' is required for GitNotification")

        if not git_issue and not git_merge_request:
            raise ValueError(
                "At least one of 'issue' or 'merge_request' is required for GitNotification"
            )

    async def push(
        self,
        message: str,
        severity: typing.Optional[
            typing.Union[mlrun.common.schemas.NotificationSeverity, str]
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Optional[typing.Union[mlrun.lists.RunList, list]] = None,
        custom_html: typing.Optional[typing.Optional[str]] = None,
        alert: typing.Optional[mlrun.common.schemas.AlertConfig] = None,
        event_data: typing.Optional[mlrun.common.schemas.Event] = None,
    ):
        git_repo = self.params.get("repo", None)
        git_issue = self.params.get("issue", None)
        git_merge_request = self.params.get("merge_request", None)
        token = (
            self.params.get("token", None)
            or self.params.get("GIT_TOKEN", None)
            or self.params.get("GITHUB_TOKEN", None)
        )
        server = self.params.get("server", None)
        gitlab = self.params.get("gitlab", False)
        await self._pr_comment(
            self._get_html(message, severity, runs, custom_html, alert, event_data),
            git_repo,
            git_issue,
            merge_request=git_merge_request,
            token=token,
            server=server,
            gitlab=gitlab,
        )

    @staticmethod
    async def _pr_comment(
        message: str,
        repo: typing.Optional[str] = None,
        issue: typing.Optional[int] = None,
        merge_request: typing.Optional[int] = None,
        token: typing.Optional[str] = None,
        server: typing.Optional[str] = None,
        gitlab: bool = False,
    ) -> str:
        """push comment message to Git system PR/issue

        :param message:  test message
        :param repo:     repo name (org/repo)
        :param issue:    pull-request/issue number
        :param token:    git system security token
        :param server:   url of the git system
        :param gitlab:   set to True for GitLab (MLRun will try to auto detect the Git system)
        :return:         pr comment id
        """
        if ("CI_PROJECT_ID" in os.environ) or (server and "gitlab" in server):
            gitlab = True
        token = (
            token
            or mlrun.get_secret_or_env("GITHUB_TOKEN")
            or mlrun.get_secret_or_env("GIT_TOKEN")
        )

        if gitlab:
            server = server or "gitlab.com"
            headers = {"PRIVATE-TOKEN": token}
            repo = repo or os.environ.get("CI_PROJECT_ID")
            # auto detect GitLab pr id from the environment
            issue = issue or os.environ.get("CI_ISSUE_IID")
            merge_request = merge_request or os.environ.get("CI_MERGE_REQUEST_IID")
            # replace slash with url encoded slash for GitLab to accept a repo name with slash
            repo = repo.replace("/", "%2F")

            if merge_request:
                url = f"https://{server}/api/v4/projects/{repo}/merge_requests/{merge_request}/notes"
            elif issue:
                url = f"https://{server}/api/v4/projects/{repo}/issues/{issue}/notes"
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "GitLab issue or merge request id not specified"
                )
        else:
            server = server or "api.github.com"
            repo = repo or os.environ.get("GITHUB_REPOSITORY")
            # auto detect pr number if not specified, in github the pr id is identified as an issue id
            # we try and read the pr (issue) id from the github actions event file/object
            if not issue and "GITHUB_EVENT_PATH" in os.environ:
                with open(os.environ["GITHUB_EVENT_PATH"]) as fp:
                    data = fp.read()
                    event = json.loads(data)
                    if "number" not in event:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"issue not found in github actions event\ndata={data}"
                        )
                    issue = event["number"]
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {token}",
            }
            url = f"https://{server}/repos/{repo}/issues/{issue}/comments"

        async with aiohttp.ClientSession() as session:
            resp = await session.post(url, headers=headers, json={"body": message})
            if not resp.ok:
                resp_text = await resp.text()
                raise mlrun.errors.MLRunBadRequestError(
                    f"Failed commenting on PR: {resp_text}"
                )
            data = await resp.json()
            return data.get("id")
