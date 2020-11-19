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
    def start(self):
        self._initialize_consumers()
        self._periodic_sync_interval_seconds = humanfriendly.parse_timespan(
            mlrun.config.config.httpdb.projects.periodic_sync_interval
        )
        self._start_periodic_sync()

    def stop(self):
        self._stop_periodic_sync()

    def ensure_project(self, session: sqlalchemy.orm.Session, name: str):
        project_names = self.list_projects(session, full=False)
        if name in project_names.projects:
            return
        logger.info(
            "Ensure project called, but project does not exist. Creating", name=name
        )
        self.create_project(session, mlrun.api.schemas.ProjectCreate(name=name))

    def create_project(
        self, session: sqlalchemy.orm.Session, project: mlrun.api.schemas.ProjectCreate
    ):
        self._run_on_all_consumers("create_project", session, project)

    def update_project(
        self,
        session: sqlalchemy.orm.Session,
        name: str,
        project: mlrun.api.schemas.ProjectUpdate,
    ):
        # ProjectUpdate allows extra fields therefore name may be there
        if hasattr(project, 'name') and name != getattr(project, 'name'):
            message = "Conflict between name in body and name in path"
            logger.warning(message, path_name=name, body_name=getattr(project, 'name'))
            raise mlrun.errors.MLRunConflictError(message)
        self._run_on_all_consumers("update_project", session, name, project)

    def delete_project(self, session: sqlalchemy.orm.Session, name: str):
        self._run_on_all_consumers("delete_project", session, name)

    def get_project(
        self, session: sqlalchemy.orm.Session, name: str
    ) -> mlrun.api.schemas.Project:
        return self._master_consumer.get_project(session, name)

    def list_projects(
        self, session: sqlalchemy.orm.Session, owner: str = None, full: bool = True,
    ) -> mlrun.api.schemas.ProjectsOutput:
        return self._master_consumer.list_projects(session, owner, full)

    def _start_periodic_sync(self):
        if self._periodic_sync_interval_seconds > 0:
            mlrun.api.utils.periodic.run_function_periodically(
                self._periodic_sync_interval_seconds,
                self._sync_projects.__name__,
                self._sync_projects,
            )

    def _stop_periodic_sync(self):
        mlrun.api.utils.periodic.cancel_periodic_function(self._sync_projects.__name__)

    def _sync_projects(self):
        session = mlrun.api.db.session.create_session()
        try:

            # preparations - build maps for easier searches
            master_projects: mlrun.api.schemas.ProjectsOutput
            consumer_projects_map: typing.Dict[str, mlrun.api.schemas.ProjectsOutput]
            master_projects, consumer_projects_map = self._run_on_all_consumers(
                "list_projects", session
            )
            master_project_names = {
                project.name for project in master_projects.projects
            }
            consumer_project_names_map = {}
            for consumer_name, consumer_projects in consumer_projects_map.items():
                consumer_project_names_map[consumer_name] = set()
                for project in consumer_projects.projects:
                    consumer_project_names_map[consumer_name].add(project.name)

            # search for projects created only in consumers
            for consumer_name, consumer_projects in consumer_projects_map.items():
                # use helper function to decrease nesting
                self._ensure_consumer_projects_synced(
                    session,
                    master_project_names,
                    consumer_project_names_map,
                    consumer_name,
                    consumer_projects,
                )

        finally:
            mlrun.api.db.session.close_session(session)

    def _ensure_consumer_projects_synced(
        self,
        session: sqlalchemy.orm.Session,
        master_project_names: typing.Set[str],
        consumer_project_names_map: typing.Dict[str, set],
        consumer_name: str,
        consumer_projects: mlrun.api.schemas.ProjectsOutput,
    ):
        for project in consumer_projects.projects:
            if project.name not in master_project_names:
                logger.debug(
                    "Found project in consumer that is not in master. Creating",
                    consumer_name=consumer_name,
                    project=project.name,
                )
                try:
                    self._master_consumer.create_project(
                        session, mlrun.api.schemas.ProjectCreate(**project.dict())
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed creating missing project in master",
                        consumer_name=consumer_name,
                        project=project.name,
                        exc=str(exc),
                    )
                else:
                    # only if we successfully created in master - align the rest of consumers
                    for (
                        _consumer_name,
                        consumer_project_names,
                    ) in consumer_project_names_map.items():
                        if project.name not in consumer_project_names:
                            logger.debug(
                                "Project is missing from consumer as well. Creating",
                                proejct_consumer_name=consumer_name,
                                missing_consumer_name=_consumer_name,
                                project=project.name,
                            )
                            try:
                                self._consumers[_consumer_name].create_project(
                                    session,
                                    mlrun.api.schemas.ProjectCreate(**project.dict()),
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Failed creating missing project in consumer",
                                    consumer_name=_consumer_name,
                                    project=project.name,
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
        self._master_consumer = self._initialize_consumer(
            mlrun.config.config.httpdb.projects.master_consumer
        )
        consumers = (
            mlrun.config.config.httpdb.projects.consumers.split(",")
            if mlrun.config.config.httpdb.projects.consumers
            else []
        )
        self._consumers = {
            consumer: self._initialize_consumer(consumer) for consumer in consumers
        }

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
