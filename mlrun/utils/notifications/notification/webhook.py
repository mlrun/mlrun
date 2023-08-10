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

import ssl
import typing

import aiohttp

import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class WebhookNotification(NotificationBase):
    """
    API/Client notification for sending run statuses in a http request
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
        url = self.params.get("url", None)
        method = self.params.get("method", "post").lower()
        headers = self.params.get("headers", {})
        override_body = self.params.get("override_body", None)
        verify_ssl = self.params.get("verify_ssl", True)

        request_body = {
            "message": message,
            "severity": severity,
            "runs": runs,
        }

        if custom_html:
            request_body["custom_html"] = custom_html

        if override_body:
            request_body = override_body

        # Create an SSL context only for HTTPS endpoints
        ssl_context = None
        if url.startswith("https"):
            ssl_context = ssl.SSLContext()
            # If verify_ssl is set to False, disable SSL certificate verification
            ssl_context.verify_mode = (
                ssl.CERT_NONE if not verify_ssl else ssl.CERT_REQUIRED
            )

        async with aiohttp.ClientSession() as session:
            response = await getattr(session, method)(
                url, headers=headers, json=request_body, ssl=ssl_context
            )
            response.raise_for_status()
