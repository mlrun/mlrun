from http import HTTPStatus

from sqlalchemy.orm import Session

from mlrun.api.constants import LogSources
from mlrun.api.api.utils import log_and_raise, log_path
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.utils import get_in, now_date, update_in


class Logs:
    @staticmethod
    def store_log(body: bytes, project: str, uid: str, append: bool = True):
        log_file = log_path(project, uid)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        with log_file.open(mode) as fp:
            fp.write(body)

    @staticmethod
    def get_log(
        db_session: Session,
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        source: LogSources = LogSources.AUTO,
    ):
        out = b""
        log_file = log_path(project, uid)
        status = None
        if log_file.exists() and source in [LogSources.AUTO, LogSources.PERSISTENCY]:
            with log_file.open("rb") as fp:
                fp.seek(offset)
                out = fp.read(size)
            status = ""
        elif source in [LogSources.AUTO, LogSources.K8S]:
            data = get_db().read_run(db_session, uid, project)
            if not data:
                log_and_raise(HTTPStatus.NOT_FOUND, project=project, uid=uid)

            status = get_in(data, "status.state", "")
            if get_k8s():
                pods = get_k8s().get_logger_pods(uid)
                if pods:
                    pod, new_status = list(pods.items())[0]
                    new_status = new_status.lower()

                    # TODO: handle in cron/tracking
                    if new_status != "pending":
                        resp = get_k8s().logs(pod)
                        if resp:
                            out = resp.encode()[offset:]
                        if status == "running":
                            now = now_date().isoformat()
                            update_in(data, "status.last_update", now)
                            if new_status == "failed":
                                update_in(data, "status.state", "error")
                                update_in(data, "status.error", "error, check logs")
                                get_db().store_run(db_session, data, uid, project)
                            if new_status == "succeeded":
                                update_in(data, "status.state", "completed")
                                get_db().store_run(db_session, data, uid, project)
                    status = new_status
                elif status == "running":
                    update_in(data, "status.state", "error")
                    update_in(data, "status.error", "pod not found, maybe terminated")
                    get_db().store_run(db_session, data, uid, project)
                    status = "failed"
        return out, status

    @staticmethod
    def get_log_mtime(project: str, uid: str) -> int:
        log_file = log_path(project, uid)
        if not log_file.exists():
            raise FileNotFoundError(f'Log file does not exist: {log_file}')
        return log_file.stat().st_mtime

    @staticmethod
    def log_file_exists(project: str, uid: str) -> bool:
        log_file = log_path(project, uid)
        return log_file.exists()
