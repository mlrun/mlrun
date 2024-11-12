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
import re
import typing
from email.message import EmailMessage

import aiosmtplib

import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers
import mlrun.utils.notifications.notification.base as base
import mlrun.utils.regex

DEFAULT_SMTP_PORT = 587


class MailNotification(base.NotificationBase):
    """
    API/Client notification for sending run statuses as a mail message
    """

    required_params = [
        "server_host",
        "server_port",
        "sender_address",
        "username",
        "password",
        "email_addresses",
    ]

    @classmethod
    def validate_params(cls, params):
        for required_param in cls.required_params:
            if required_param not in params:
                raise ValueError(
                    f"Parameter '{required_param}' is required for MailNotification"
                )

        if not isinstance(params["email_addresses"], (str, list)):
            raise ValueError(
                "Parameter 'email_addresses' must be a string or a list of strings"
            )
        email_addresses = params["email_addresses"]
        if isinstance(params["email_addresses"], str):
            email_addresses = params["email_addresses"].split(",")

        for email_address in email_addresses:
            if not isinstance(email_address, str):
                raise ValueError(
                    "Parameter 'email_addresses' must be a string or a list of strings"
                )

            if not re.match(mlrun.utils.regex.mail_regex, email_address):
                raise ValueError(
                    f"Invalid email address '{email_address}' in 'email_addresses'"
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
        self.params.setdefault("subject", f"[{severity}] {message}")
        self.params.setdefault("body", message)
        await self._send_email(**self.params)

    @staticmethod
    async def _send_email(
        email_addresses: str,
        sender_address: str,
        server_host: str,
        server_port: int,
        username: str,
        password: str,
        use_tls: bool,
        start_tls: bool,
        validate_certs: bool,
        subject: str,
        body: str,
        **kwargs,
    ):
        # Create the email message
        message = EmailMessage()
        message["From"] = sender_address
        message["To"] = email_addresses
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
            validate_certs=validate_certs,
            start_tls=start_tls,
        )

    @classmethod
    def enrich_default_params(
        cls, params: dict, default_params: typing.Optional[dict] = None
    ) -> dict:
        params = super().enrich_default_params(params, default_params)

        cls._load_json_from_str_param_with_default_json_value(params, "use_tls", "true")
        cls._load_json_from_str_param_with_default_json_value(
            params, "start_tls", "false"
        )
        cls._load_json_from_str_param_with_default_json_value(
            params, "validate_certs", "true"
        )

        params.setdefault("server_port", DEFAULT_SMTP_PORT)

        default_mail_address = params.pop("default_email_addresses", "")
        email_addresses = params.get("email_addresses", default_mail_address)
        if isinstance(email_addresses, list):
            email_addresses = ",".join(email_addresses)
        params["email_addresses"] = email_addresses

        return params
