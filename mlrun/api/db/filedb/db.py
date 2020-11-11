from typing import List, Any, Dict

from mlrun.api import schemas
from mlrun.api.db.base import DBError
from mlrun.api.db.base import DBInterface
from mlrun.db.base import RunDBError
from mlrun.db.filedb import FileRunDB


class FileDB(DBInterface):
    def __init__(self, dirpath="", format=".yaml"):
        self.db = FileRunDB(dirpath, format)

    def initialize(self, session):
        self.db.connect()

    def store_log(self, session, uid, project="", body=None, append=False):
        return self._transform_run_db_error(
            self.db.store_log, uid, project, body, append
        )

    def get_log(self, session, uid, project="", offset=0, size=0):
        return self._transform_run_db_error(self.db.get_log, uid, project, offset, size)

    def store_run(self, session, struct, uid, project="", iter=0):
        return self._transform_run_db_error(
            self.db.store_run, struct, uid, project, iter
        )

    def update_run(self, session, updates: dict, uid, project="", iter=0):
        return self._transform_run_db_error(
            self.db.update_run, updates, uid, project, iter
        )

    def read_run(self, session, uid, project="", iter=0):
        return self._transform_run_db_error(self.db.read_run, uid, project, iter)

    def list_runs(
        self,
        session,
        name="",
        uid=None,
        project="",
        labels=None,
        state="",
        sort=True,
        last=0,
        iter=False,
    ):
        return self._transform_run_db_error(
            self.db.list_runs, name, uid, project, labels, state, sort, last, iter
        )

    def del_run(self, session, uid, project="", iter=0):
        return self._transform_run_db_error(self.db.del_run, uid, project, iter)

    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        return self._transform_run_db_error(
            self.db.del_runs, name, project, labels, state, days_ago
        )

    def store_artifact(
        self, session, key, artifact, uid, iter=None, tag="", project=""
    ):
        return self._transform_run_db_error(
            self.db.store_artifact, key, artifact, uid, iter, tag, project
        )

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        return self._transform_run_db_error(
            self.db.read_artifact, key, tag, iter, project
        )

    def list_artifacts(
        self,
        session,
        name="",
        project="",
        tag="",
        labels=None,
        since=None,
        until=None,
        kind=None,
        category: schemas.ArtifactCategories = None,
    ):
        return self._transform_run_db_error(
            self.db.list_artifacts, name, project, tag, labels, since, until
        )

    def del_artifact(self, session, key, tag="", project=""):
        return self._transform_run_db_error(self.db.del_artifact, key, tag, project)

    def del_artifacts(self, session, name="", project="", tag="", labels=None):
        return self._transform_run_db_error(
            self.db.del_artifacts, name, project, tag, labels
        )

    def store_function(
        self, session, function, name, project="", tag="", versioned=False
    ):
        return self._transform_run_db_error(
            self.db.store_function, function, name, project, tag, versioned
        )

    def get_function(self, session, name, project="", tag="", hash_key=""):
        return self._transform_run_db_error(
            self.db.get_function, name, project, tag, hash_key
        )

    def delete_function(self, session, project: str, name: str):
        raise NotImplementedError()

    def list_functions(self, session, name=None, project="", tag="", labels=None):
        return self._transform_run_db_error(
            self.db.list_functions, name, project, tag, labels
        )

    def store_schedule(self, session, data):
        return self._transform_run_db_error(self.db.store_schedule, data)

    def list_projects(self, session, owner=None):
        return self._transform_run_db_error(self.db.list_projects)

    def add_project(self, session, project: dict):
        raise NotImplementedError()

    def update_project(self, session, name, data: dict):
        raise NotImplementedError()

    def get_project(self, session, name=None, project_id=None):
        raise NotImplementedError()

    def delete_project(self, session, name: str):
        raise NotImplementedError()

    def create_feature_set(
        self, session, project, feature_set: schemas.FeatureSet, versioned=True
    ):
        raise NotImplementedError()

    def get_feature_set(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> schemas.FeatureSet:
        raise NotImplementedError()

    def list_features(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeaturesOutput:
        raise NotImplementedError()

    def list_feature_sets(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeatureSetsOutput:
        raise NotImplementedError()

    def update_feature_set(
        self,
        session,
        project,
        name,
        feature_set_update: schemas.FeatureSetUpdate,
        tag=None,
        uid=None,
    ):
        raise NotImplementedError()

    def delete_feature_set(self, session, project, name):
        raise NotImplementedError()

    def list_artifact_tags(self, session, project):
        return self._transform_run_db_error(self.db.list_artifact_tags, project)

    def create_schedule(
        self,
        session,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
        labels: Dict = None,
    ):
        raise NotImplementedError()

    def update_schedule(
        self,
        session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: schemas.ScheduleCronTrigger = None,
        labels: Dict = None,
        last_run_uri: str = None,
    ):
        raise NotImplementedError()

    def list_schedules(
        self,
        session,
        project: str = None,
        name: str = None,
        labels: str = None,
        kind: schemas.ScheduleKinds = None,
    ) -> List[schemas.ScheduleRecord]:
        raise NotImplementedError()

    def get_schedule(self, session, project: str, name: str) -> schemas.ScheduleRecord:
        raise NotImplementedError()

    def delete_schedule(self, session, project: str, name: str):
        raise NotImplementedError()

    @staticmethod
    def _transform_run_db_error(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RunDBError as exc:
            raise DBError(exc.args)
