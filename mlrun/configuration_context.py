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

import contextvars
from typing import Optional

# Context storage for MLRunConfigurationContext
mlrun_configuration_context: contextvars.ContextVar[
    Optional["MLRunConfigurationContext"]
] = contextvars.ContextVar("mlrun_configuration_context", default=None)


class MLRunConfigurationContext:
    """
    Context manager for MLRun configuration options.
    Settings here override any function-level configuration.

    Example::

        with mlrun.MLRunConfigurationContext(auth_token_name="my-token"):
            func.run()
            project.enable_model_monitoring()

    :param auth_token_name: Name of the authentication token to use for operations.
    """

    __slots__ = ("auth_token_name", "_token")

    def __init__(self, auth_token_name: Optional[str] = None):
        self.auth_token_name = auth_token_name
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self._token = mlrun_configuration_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlrun_configuration_context.reset(self._token)
        return False

    def __repr__(self) -> str:
        return f"MLRunConfigurationContext(auth_token_name={self.auth_token_name!r})"

    @staticmethod
    def get_auth_token_name() -> Optional[str]:
        """
        Get the auth token name from the current configuration context, if any.

        :return: The auth token name if set in the current context, None otherwise.
        """
        ctx = mlrun_configuration_context.get()
        return ctx.auth_token_name if ctx else None
