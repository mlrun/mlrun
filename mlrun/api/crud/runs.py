import typing

import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.config
import mlrun.errors
import mlrun.lists
import mlrun.runtimes
import mlrun.runtimes.constants
import mlrun.utils.singleton
from mlrun.utils import logger


class Runs(metaclass=mlrun.utils.singleton.Singleton,):
    def store_run(
        self,
        db_session: sqlalchemy.orm.Session,
        data: dict,
        uid: str,
        iter: int = 0,
        project: str = mlrun.mlconf.default_project,
    ):
        project = project or mlrun.mlconf.default_project
        logger.info("Storing run", data=data)
        mlrun.api.utils.singletons.db.get_db().store_run(
            db_session, data, uid, project, iter=iter,
        )

    def update_run(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        uid: str,
        iter: int,
        data: dict,
    ):
        project = project or mlrun.mlconf.default_project
        logger.debug("Updating run", project=project, uid=uid, iter=iter, data=data)
        # TODO: do some desired state for run, it doesn't make sense that API user changes the status in order to
        #  trigger abortion
        if (
            data
            and data.get("status.state") == mlrun.runtimes.constants.RunStates.aborted
        ):
            current_run = mlrun.api.utils.singletons.db.get_db().read_run(
                db_session, uid, project, iter
            )
            if (
                current_run.get("status", {}).get("state")
                in mlrun.runtimes.constants.RunStates.terminal_states()
            ):
                raise mlrun.errors.MLRunConflictError(
                    "Run is already in terminal state, can not be aborted"
                )
            runtime_kind = current_run.get("metadata", {}).get("labels", {}).get("kind")
            if runtime_kind not in mlrun.runtimes.RuntimeKinds.abortable_runtimes():
                raise mlrun.errors.MLRunBadRequestError(
                    f"Run of kind {runtime_kind} can not be aborted"
                )
            # aborting the run meaning deleting its runtime resources
            # TODO: runtimes crud interface should ideally expose some better API that will hold inside itself the
            #  "knowledge" on the label selector
            mlrun.api.crud.RuntimeResources().delete_runtime_resources(
                db_session,
                label_selector=f"mlrun/project={project},mlrun/uid={uid}",
                force=True,
            )
        mlrun.api.utils.singletons.db.get_db().update_run(
            db_session, data, uid, project, iter
        )

    def get_run(
        self,
        db_session: sqlalchemy.orm.Session,
        uid: str,
        iter: int,
        project: str = mlrun.mlconf.default_project,
    ) -> dict:
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().read_run(
            db_session, uid, project, iter
        )

    def list_runs(
        self,
        db_session: sqlalchemy.orm.Session,
        name=None,
        uid=None,
        project: str = mlrun.mlconf.default_project,
        labels=None,
        states: typing.Optional[typing.List[str]] = None,
        sort=True,
        last=0,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
        partition_by: mlrun.api.schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.api.schemas.SortField = None,
        partition_order: mlrun.api.schemas.OrderType = mlrun.api.schemas.OrderType.desc,
    ):
        project = project or mlrun.mlconf.default_project
        return mlrun.api.utils.singletons.db.get_db().list_runs(
            db_session,
            name,
            uid,
            project,
            labels,
            states,
            sort,
            last,
            iter,
            start_time_from,
            start_time_to,
            last_update_time_from,
            last_update_time_to,
            partition_by,
            rows_per_partition,
            partition_sort_by,
            partition_order,
        )

    def delete_run(
        self,
        db_session: sqlalchemy.orm.Session,
        uid: str,
        iter: int,
        project: str = mlrun.mlconf.default_project,
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.singletons.db.get_db().del_run(db_session, uid, project, iter)

    def delete_runs(
        self,
        db_session: sqlalchemy.orm.Session,
        name=None,
        project: str = mlrun.mlconf.default_project,
        labels=None,
        state=None,
        days_ago: int = 0,
    ):
        project = project or mlrun.mlconf.default_project
        mlrun.api.utils.singletons.db.get_db().del_runs(
            db_session, name, project, labels, state, days_ago
        )
