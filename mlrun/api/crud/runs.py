import typing

import sqlalchemy.orm

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.errors
import mlrun.runtimes
import mlrun.runtimes.constants
import mlrun.utils.singleton
from mlrun.utils import logger


class Runs(metaclass=mlrun.utils.singleton.Singleton,):
    def update_run(
        self,
        session: sqlalchemy.orm.Session,
        project: str,
        uid: str,
        iter: int,
        data: dict,
        leader_session: typing.Optional[str] = None,
    ):
        logger.debug("Updating run", project=project, uid=uid, iter=iter, data=data)
        # TODO: do some desired state for run, it doesn't make sense that API user changes the status in order to
        #  trigger abortion
        if (
            data
            and data.get("status.state") == mlrun.runtimes.constants.RunStates.aborted
        ):
            current_run = mlrun.api.utils.singletons.db.get_db().read_run(
                session, uid, project, iter
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
            mlrun.api.crud.Runtimes().delete_runtimes(
                session,
                label_selector=f"mlrun/project={project},mlrun/uid={uid}",
                force=True,
                leader_session=leader_session,
            )
        mlrun.api.utils.singletons.db.get_db().update_run(
            session, data, uid, project, iter
        )
