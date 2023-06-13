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

import igz_mgmt.schemas.manual_events
import semver

import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.events.base
import mlrun.common.schemas
from mlrun.utils import logger

PROJECT_AUTH_SECRET_CREATED = "Software.Project.AuthSecret.Created"
PROJECT_AUTH_SECRET_UPDATED = "Software.Project.AuthSecret.Updated"
PROJECT_SECRET_CREATED = "Software.Project.Secret.Created"
PROJECT_SECRET_UPDATED = "Software.Project.Secret.Updated"
PROJECT_SECRET_DELETED = "Software.Project.Secret.Deleted"


class Client(mlrun.api.utils.events.base.BaseEventClient):
    def __init__(self, access_key: str = None, verbose: bool = None):
        self.access_key = (
            access_key
            or mlrun.mlconf.events.access_key
            or mlrun.mlconf.get_v3io_access_key()
        )
        self.verbose = verbose if verbose is not None else mlrun.mlconf.events.verbose
        self.source = "mlrun-api"

    def emit(self, event: igz_mgmt.schemas.manual_events.ManualEventSchema):
        try:
            mlrun.api.utils.clients.iguazio.Client().emit_manual_event(
                self.access_key, event
            )
        except Exception as exc:
            if self.verbose:
                logger.warning(
                    "Failed to emit event",
                    event=event,
                    exc_info=exc,
                )

    def generate_project_auth_secret_event(
        self,
        username: str,
        secret_name: str,
        action: mlrun.common.schemas.AuthSecretEventActions,
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        """
        Generate a project auth secret event
        :param username:  username
        :param secret_name:  secret name
        :param action: preformed action
        :return: event object to emit
        """
        if action == mlrun.common.schemas.SecretEventActions.created:
            return self.generate_project_auth_secret_created_event(
                username, secret_name
            )
        elif action == mlrun.common.schemas.SecretEventActions.updated:
            return self.generate_project_auth_secret_updated_event(
                username, secret_name
            )
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported action {action}")

    def generate_project_auth_secret_created_event(
        self, username: str, secret_name: str
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.4-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_AUTH_SECRET_CREATED,
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="username", value=username
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.4-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_AUTH_SECRET_CREATED,
                description=f"User {username} created secret {secret_name}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_auth_secret_updated_event(
        self, username: str, secret_name: str
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.4-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_AUTH_SECRET_UPDATED,
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="username", value=username
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.4-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_AUTH_SECRET_UPDATED,
                description=f"User {username} updated secret {secret_name}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: typing.List[str] = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        """
        Generate a project secret event
        :param project: project name
        :param secret_name: secret name
        :param secret_keys: secret keys, optional, only relevant for created/updated events
        :param action: preformed action
        :return: event object to emit
        """
        if action == "created":
            return self.generate_project_secret_created_event(
                project, secret_name, secret_keys
            )
        elif action == "updated":
            return self.generate_project_secret_updated_event(
                project, secret_name, secret_keys
            )
        elif action == "deleted":
            return self.generate_project_secret_deleted_event(project, secret_name)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported action {action}")

    def generate_project_secret_created_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        normalized_secret_keys = self._list_to_string(secret_keys)
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.4-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_CREATED,
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_keys", value=normalized_secret_keys
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.4-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_CREATED,
                description=f"Created project secret {secret_name} with secret keys {normalized_secret_keys}"
                f" for project {project}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_secret_updated_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: typing.List[str],
        updated: bool = True,
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        action = "Updated" if updated else "Deleted"
        normalized_secret_keys = self._list_to_string(secret_keys)
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.4-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_UPDATED,
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_keys", value=normalized_secret_keys
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="action", value=action
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.4-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_UPDATED,
                description=f"{action} secret keys {normalized_secret_keys} of project secret {secret_name} "
                f"for project {project}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_secret_deleted_event(self, project: str, secret_name: str):
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.4-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_DELETED,
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.4-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind=PROJECT_SECRET_DELETED,
                description=f"Deleted project secret {secret_name} for project {project}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    @staticmethod
    def _list_to_string(list_to_convert: typing.List[str]) -> str:
        return ", ".join(list_to_convert)
