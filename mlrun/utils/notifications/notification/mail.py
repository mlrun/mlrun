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
import typing
from email.message import EmailMessage

import aiosmtplib

import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class MailNotification(NotificationBase):
    """
    API/Client notification for sending run statuses as a mail message
    """

    required_params = [
        "server_host",
        "server_port",
        "sender_address",
        "username",
        "password",
        "to_email",
    ]

    @classmethod
    def validate_params(cls, params):
        for required_param in cls.required_params:
            if required_param not in params:
                raise ValueError(
                    f"Parameter '{required_param}' is required for MailNotification"
                )

    async def push(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
        alert: mlrun.common.schemas.AlertConfig = None,
        event_data: mlrun.common.schemas.Event = None,
    ):
        await self._send_async_email(body=message, **self.params)

    @staticmethod
    async def _send_async_email(
        to_email,
        sender_address,
        server_host,
        server_port,
        username,
        password,
        use_tls,
        subject="",
        body="",
        **kwargs,
    ):
        # Create the email message
        message = EmailMessage()
        message["From"] = sender_address
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        # Send the email
        await aiosmtplib.send(
            message,
            hostname=server_host,
            port=server_port,
            username=username,
            password=password,
            use_tls=use_tls,
            validate_certs=use_tls,
        )

    @classmethod
    def fill_default_params(cls, params: dict, default_params: dict):
        params = super().fill_default_params(params, default_params)

        if type(params["use_tls"]) is str:
            params["use_tls"] = json.loads(params.get("use_tls", "true"))

        default_mail_address = params.pop("default_email_address", None)
        params.setdefault("to_email", default_mail_address)
        return params
