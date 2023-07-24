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
import typing

import mlrun.api.utils.events.base as base_events
import mlrun.common.schemas


class NopClient(base_events.BaseEventClient):
    def emit(self, event):
        return

    def generate_auth_secret_event(
        self,
        username: str,
        secret_name: str,
        action: mlrun.common.schemas.AuthSecretEventActions,
    ):
        """
        Generate an auth secret event
        :param username:  username
        :param secret_name:  secret name
        :param action: preformed action
        :return: event object to emit
        """
        return

    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: typing.List[str] = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
    ):
        """
        Generate a project secret event
        :param project: project name
        :param secret_name: secret name
        :param secret_keys: secret keys, optional, only relevant for created/updated events
        :param action: preformed action
        :return: event object to emit
        """
        pass
