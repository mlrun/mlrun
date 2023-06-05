import typing

import igz_mgmt.schemas.manual_events
import semver

import mlrun.api.utils.clients.iguazio
import mlrun.api.utils.events.base


class Client(mlrun.api.utils.events.base.BaseEventClient):
    def __init__(self, access_key: str = None):
        self.access_key = access_key or mlrun.mlconf.get_v3io_access_key()
        self.source = "mlrun-api"

    def emit(self, event: igz_mgmt.schemas.manual_events.ManualEventSchema):
        mlrun.api.utils.clients.iguazio.Client().emit_manual_event(
            self.access_key, event
        )

    def generate_project_auth_secret_created_event(
        self, username: str, secret_name: str
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.3-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.AuthSecret.Created",
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="username", value=username
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.3-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.AuthSecret.Created",
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
            "3.5.3-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.AuthSecret.Updated",
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="username", value=username
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.3-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.AuthSecret.Updated",
                description=f"User {username} updated secret {secret_name}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_secret_created_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ) -> igz_mgmt.schemas.manual_events.ManualEventSchema:
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.3-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secret.Created",
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_keys", value=", ".join(secret_keys)
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.3-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secret.Created",
                description=f"Created project secret {secret_name} with secret keys {secret_keys}"
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
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.3-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secret.Updated",
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_keys", value=", ".join(secret_keys)
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="action", value=action
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.3-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secret.Updated",
                description=f"{action} secret keys {secret_keys} of project secret {secret_name} for project {project}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )

    def generate_project_secrets_deleted_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ):
        # adding condition as old iguazio versions doesn't contain the configured events, therefore we need to
        # specify a more detailed event
        if mlrun.mlconf.get_parsed_igz_version() >= semver.VersionInfo.parse(
            "3.5.3-b1"
        ):
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secrets.Deleted",
                parameters_text=[
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="project", value=project
                    ),
                    igz_mgmt.schemas.manual_events.ParametersText(
                        name="secret_name", value=secret_name
                    ),
                ],
            )
        # TODO: remove this else when we drop support for iguazio < 3.5.3-b1
        else:
            return igz_mgmt.schemas.manual_events.ManualEventSchema(
                source=self.source,
                kind="Software.Project.Secrets.Deleted",
                description=f"Deleted project secret {secret_name} for project {project}",
                severity=igz_mgmt.schemas.manual_events.EventSeverity.info,
                classification=igz_mgmt.schemas.manual_events.EventClassification.audit,
                system_event=False,
                visibility=igz_mgmt.schemas.manual_events.EventVisibility.external,
            )
