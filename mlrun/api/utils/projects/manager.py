import collections
import typing

import humanfriendly
import sqlalchemy.orm

import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.periodic
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.projects.consumers.nop
import mlrun.api.utils.projects.consumers.nuclio
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.singleton
from mlrun.utils import logger


class ProjectsManager(metaclass=mlrun.utils.singleton.Singleton):
    def initialize(self):
        logger.info("Initializing projects manager")
        self._initialize_consumers()
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.config.config.httpdb.projects.periodic_sync_interval
        )
        # run one sync to start off on the right foot
        self._sync_projects()
        self._start_periodic_sync()

    def shutdown(self):
        logger.info("Shutting down projects manager")
        self._stop_periodic_sync()

    def ensure_project(self, session: sqlalchemy.orm.Session, name: str):
        project_names = self.list_projects(
            session, format_=mlrun.api.schemas.Format.name_only
        )
        if name in project_names.projects:
            return
        logger.info(
            "Ensure project called, but project does not exist. Creating", name=name
        )
        self.create_project(session, mlrun.api.schemas.Project(name=name))

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.Project
    ) -> mlrun.api.schemas.Project:
        self._run_on_all_consumers("create_project", session, project)
        return self.get_project(session, project.name)

    def store_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.Project,
    ):
        self._validate_body_and_path_names_matches(name, project)
        self._run_on_all_consumers("store_project", session, name, project)
        return self.get_project(session, name)

    def patch_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectPatch,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ):
        self._validate_body_and_path_names_matches(name, project)
        self._run_on_all_consumers("patch_project", session, name, project, patch_mode)
        return self.get_project(session, name)

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        self._run_on_all_consumers("delete_project", session, name)

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return self._master_consumer.get_project(session, name)

    def list_projects(
        self,
        session: sqlalchemy.orm.Session,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return self._master_consumer.list_projects(session, owner, format_)

    def _start_periodic_sync(self):
        # if no consumers no need for sync
        # the > 0 condition is to allow ourselves to disable the sync fomr configuration
        if self._periodic_sync_interval_seconds > 0 and self._consumers:
            logger.info(
                "Starting periodic projects sync",
                interval=self._periodic_sync_interval_seconds,
            )
            mlrun.api.utils.periodic.run_function_periodically(
                self._periodic_sync_interval_seconds,
                self._sync_projects.__name__,
                False,
                self._sync_projects,
            )

    def _stop_periodic_sync(self):
        mlrun.api.utils.periodic.cancel_periodic_function(self._sync_projects.__name__)

    def _sync_projects(self):
        session = mlrun.api.db.session.create_session()
        try:
            # re-generating all of the maps every time since _ensure_consumer_projects_synced might cause changes
            master_projects: mlrun.api.schemas.ProjectsOutput
            consumer_projects_map: typing.Dict[str, mlrun.api.schemas.ProjectsOutput]
            master_projects, consumer_projects_map = self._run_on_all_consumers(
                "list_projects", session
            )
            master_project_names = {
                project.name for project in master_projects.projects
            }
            # create reverse map project -> consumer names
            project_consumer_names_map = collections.defaultdict(set)
            for _consumer_name, consumer_projects in consumer_projects_map.items():
                for project in consumer_projects.projects:
                    project_consumer_names_map[project.name].add(_consumer_name)

            # create map - consumer name -> project name -> project for easier searches
            consumers_projects_map = collections.defaultdict(dict)
            for _consumer_name, consumer_projects in consumer_projects_map.items():
                for project in consumer_projects.projects:
                    consumers_projects_map[_consumer_name][project.name] = project

            # create map - master project name -> master project for easier searches
            master_projects_map = {}
            for master_project in master_projects.projects:
                master_projects_map[master_project.name] = master_project

            all_project = master_project_names.copy()
            all_project.update(project_consumer_names_map.keys())

            for project in all_project:
                self._ensure_project_synced(
                    session,
                    master_project_names,
                    project_consumer_names_map[project],
                    project,
                    consumers_projects_map,
                    master_projects_map,
                )
        finally:
            mlrun.api.db.session.close_session(session)

    def _ensure_project_synced(
        self,
        session: sqlalchemy.orm.Session,
        master_project_names: typing.Set[str],
        consumer_names: typing.Set[str],
        project_name: str,
        consumers_projects_map: typing.Dict[
            str, typing.Dict[str, mlrun.api.schemas.Project]
        ],
        master_projects_map: typing.Dict[str, mlrun.api.schemas.Project],
    ):
        # FIXME: This function only handles syncing project existence, i.e. if a user updates a project attribute
        #  through one of the consumers this change won't be synced and the projects will be left with this discrepancy
        #  for ever
        project = None
        project_consumer_name = None
        # first verify that the master have this project
        if project_name not in master_project_names:
            project_in_master = False
            logger.debug(
                "Found project in some of the consumers that is not in master. Creating",
                consumer_names=consumer_names,
                project=project_name,
            )
            try:
                # Heuristically pick the first consumer
                project_consumer_name = list(consumer_names)[0]
                project = consumers_projects_map[project_consumer_name][project_name]
                self._master_consumer.create_project(
                    session, mlrun.api.schemas.Project(**project.dict())
                )
            except Exception as exc:
                logger.warning(
                    "Failed creating missing project in master",
                    project_consumer_name=project_consumer_name,
                    project=project,
                    project_name=project_name,
                    exc=str(exc),
                )
            else:
                project_in_master = True
        else:
            project_in_master = True
            project = master_projects_map[project_name]

        # only if project in master - align the rest of consumers
        if project_in_master:
            missing_consumers = set(consumer_names).symmetric_difference(
                self._consumers.keys()
            )
            if missing_consumers:
                for missing_consumer in missing_consumers:
                    logger.debug(
                        "Project is missing from consumer. Creating",
                        missing_consumer_name=missing_consumer,
                        project_consumer_name=project_consumer_name,
                        project_name=project_name,
                        project=project,
                    )
                    try:
                        self._consumers[missing_consumer].create_project(
                            session, mlrun.api.schemas.Project(**project.dict()),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed creating missing project in consumer",
                            missing_consumer_name=missing_consumer,
                            project_consumer_name=project_consumer_name,
                            project_name=project_name,
                            project=project,
                            exc=str(exc),
                        )

    def _run_on_all_consumers(
        self, method: str, *args, **kwargs
    ) -> typing.Tuple[typing.Any, typing.Dict[str, typing.Any]]:
        # TODO: do it concurrently
        master_response = getattr(self._master_consumer, method)(*args, **kwargs)
        consumer_responses = {
            consumer_name: getattr(consumer, method)(*args, **kwargs)
            for consumer_name, consumer in self._consumers.items()
        }
        return master_response, consumer_responses

    def _initialize_consumers(self):
        master_name = mlrun.config.config.httpdb.projects.master_consumer
        self._master_consumer = self._initialize_consumer(master_name)
        consumers = (
            mlrun.config.config.httpdb.projects.consumers.split(",")
            if mlrun.config.config.httpdb.projects.consumers
            else []
        )
        self._consumers = {
            consumer: self._initialize_consumer(consumer) for consumer in consumers
        }
        logger.debug(
            "Initialized master and consumers",
            master=master_name,
            consumers=list(self._consumers.keys()),
        )

    def _initialize_consumer(
        self, name: str
    ) -> mlrun.api.utils.projects.consumers.base.Consumer:
        # importing here to avoid circular import (db using project manager using mlrun consumer using db)
        import mlrun.api.utils.projects.consumers.mlrun

        consumers_classes_map = {
            "mlrun": mlrun.api.utils.projects.consumers.mlrun.Consumer,
            "nuclio": mlrun.api.utils.projects.consumers.nuclio.Consumer,
            # for tests
            "nop": mlrun.api.utils.projects.consumers.nop.Consumer,
            "nop2": mlrun.api.utils.projects.consumers.nop.Consumer,
        }
        if name not in consumers_classes_map:
            raise ValueError(f"Unknown consumer name: {name}")
        return consumers_classes_map[name]()

    @staticmethod
    def _validate_body_and_path_names_matches(
        name: str, project: mlrun.api.schemas.ProjectPatch
    ):
        # ProjectPatch allow extra fields, therefore although it doesn't have name in the schema, name might be there
        if hasattr(project, "name") and name != getattr(project, "name"):
            message = "Conflict between name in body and name in path"
            logger.warning(message, path_name=name, body_name=getattr(project, "name"))
            raise mlrun.errors.MLRunConflictError(message)
