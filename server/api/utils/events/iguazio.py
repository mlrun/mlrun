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

import igz_mgmt.schemas.events

import mlrun.common.schemas
import server.api.utils.clients.iguazio
import server.api.utils.events.base as base_events
from mlrun.utils import logger

PROJECT_AUTH_SECRET_CREATED = "Security.Project.AuthSecret.Created"
PROJECT_AUTH_SECRET_UPDATED = "Security.Project.AuthSecret.Updated"
PROJECT_SECRET_CREATED = "Security.Project.Secret.Created"
PROJECT_SECRET_UPDATED = "Security.Project.Secret.Updated"
PROJECT_SECRET_DELETED = "Security.Project.Secret.Deleted"


class Client(base_events.BaseEventClient):
    def __init__(self, access_key: str = None, verbose: bool = None):
        self.access_key = (
            access_key
            or mlrun.mlconf.events.access_key
            or mlrun.mlconf.get_v3io_access_key()
        )
        self.verbose = verbose if verbose is not None else mlrun.mlconf.events.verbose
        self.source = "mlrun-api"

    def emit(self, event: igz_mgmt.Event):
        try:
            logger.debug("Emitting event", event=event)
            server.api.utils.clients.iguazio.Client().emit_manual_event(
                self.access_key, event
            )
        except Exception as exc:
            logger.warning(
                "Failed to emit event",
                event=event,
                exc_info=exc,
            )

    def generate_auth_secret_event(
        self,
        username: str,
        secret_name: str,
        action: mlrun.common.schemas.AuthSecretEventActions,
    ) -> igz_mgmt.AuditEvent:
        """
        Generate an auth secret event
        :param username:        username
        :param secret_name:     secret name
        :param action:          preformed action
        :return: event object to emit
        """
        if action in [
            mlrun.common.schemas.SecretEventActions.created,
            mlrun.common.schemas.SecretEventActions.updated,
        ]:
            return self._generate_auth_secret_event(username, secret_name, action)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported action {action}")

    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: list[str] = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
    ) -> igz_mgmt.AuditEvent:
        """
        Generate a project secret event
        :param project:     project name
        :param secret_name: secret name
        :param secret_keys: secret keys, optional, only relevant for created/updated events
        :param action:      preformed action
        :return: event object to emit
        """
        if action == mlrun.common.schemas.SecretEventActions.created:
            return self._generate_project_secret_created_event(
                project, secret_name, secret_keys
            )
        elif action == mlrun.common.schemas.SecretEventActions.updated:
            return self._generate_project_secret_updated_event(
                project, secret_name, secret_keys
            )
        elif action == mlrun.common.schemas.SecretEventActions.deleted:
            return self._generate_project_secret_deleted_event(project, secret_name)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported action {action}")

    def _generate_auth_secret_event(
        self, username: str, secret_name: str, action: str
    ) -> igz_mgmt.AuditEvent:
        return igz_mgmt.AuditEvent(
            source=self.source,
            kind=PROJECT_AUTH_SECRET_CREATED,
            description=f"User {username} {action} secret {secret_name}",
            parameters_text=[
                igz_mgmt.schemas.events.ParametersText(name="username", value=username),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_name", value=secret_name
                ),
            ],
            severity=igz_mgmt.constants.EventSeverity.info,
            classification=igz_mgmt.constants.EventClassification.security,
            system_event=True,
            visibility=igz_mgmt.constants.EventVisibility.external,
        )

    def _generate_project_secret_created_event(
        self, project: str, secret_name: str, secret_keys: list[str]
    ) -> igz_mgmt.AuditEvent:
        normalized_secret_keys = self._list_to_string(secret_keys)
        return igz_mgmt.AuditEvent(
            source=self.source,
            kind=PROJECT_SECRET_CREATED,
            parameters_text=[
                igz_mgmt.schemas.events.ParametersText(
                    name="project_name", value=project
                ),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_name", value=secret_name
                ),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_keys", value=normalized_secret_keys
                ),
            ],
            description=f"Project {project} secret created",
            severity=igz_mgmt.constants.EventSeverity.info,
            classification=igz_mgmt.constants.EventClassification.security,
            system_event=True,
            visibility=igz_mgmt.constants.EventVisibility.external,
        )

    def _generate_project_secret_updated_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: list[str],
    ) -> igz_mgmt.AuditEvent:
        normalized_secret_keys = self._list_to_string(secret_keys)
        return igz_mgmt.AuditEvent(
            source=self.source,
            kind=PROJECT_SECRET_UPDATED,
            description=f"Project {project} secret updated",
            parameters_text=[
                igz_mgmt.schemas.events.ParametersText(
                    name="project_name", value=project
                ),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_name", value=secret_name
                ),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_keys", value=normalized_secret_keys
                ),
            ],
            severity=igz_mgmt.constants.EventSeverity.info,
            classification=igz_mgmt.constants.EventClassification.security,
            system_event=True,
            visibility=igz_mgmt.constants.EventVisibility.external,
        )

    def _generate_project_secret_deleted_event(
        self, project: str, secret_name: str
    ) -> igz_mgmt.AuditEvent:
        return igz_mgmt.AuditEvent(
            source=self.source,
            kind=PROJECT_SECRET_DELETED,
            description=f"Project {project} secret deleted",
            parameters_text=[
                igz_mgmt.schemas.events.ParametersText(
                    name="project_name", value=project
                ),
                igz_mgmt.schemas.events.ParametersText(
                    name="secret_name", value=secret_name
                ),
            ],
            severity=igz_mgmt.constants.EventSeverity.info,
            classification=igz_mgmt.constants.EventClassification.security,
            system_event=True,
            visibility=igz_mgmt.constants.EventVisibility.external,
        )

    @staticmethod
    def _list_to_string(list_to_convert: list[str]) -> str:
        return ", ".join(list_to_convert)
