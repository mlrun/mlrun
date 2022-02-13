from typing import Any, Dict, List, Optional, Tuple

from mlrun.api import schemas
from mlrun.api.db.base import DBError, DBInterface
from mlrun.db.base import RunDBError
from mlrun.db.filedb import FileRunDB


class FileDB(DBInterface):
    def __init__(self, dirpath="", format=".yaml"):
        self.db = FileRunDB(dirpath, format)

    def initialize(self, session):
        self.db.connect()

    def store_log(
        self, session, uid, project="", body=None, append=False,
    ):
        return self._transform_run_db_error(
            self.db.store_log, uid, project, body, append
        )

    def get_log(self, session, uid, project="", offset=0, size=0):
        return self._transform_run_db_error(self.db.get_log, uid, project, offset, size)

    def store_run(
        self, session, struct, uid, project="", iter=0,
    ):
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
        states=None,
        sort=True,
        last=0,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
        partition_by: schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ):
        return self._transform_run_db_error(
            self.db.list_runs,
            name,
            uid,
            project,
            labels,
            states[0] if states else "",
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

    def del_run(self, session, uid, project="", iter=0):
        return self._transform_run_db_error(self.db.del_run, uid, project, iter)

    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        return self._transform_run_db_error(
            self.db.del_runs, name, project, labels, state, days_ago
        )

    def store_artifact(
        self, session, key, artifact, uid, iter=None, tag="", project="",
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
        iter: int = None,
        best_iteration: bool = False,
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
        self, session, function, name, project="", tag="", versioned=False,
    ) -> str:
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

    def generate_projects_summaries(
        self, session, projects: List[str]
    ) -> List[schemas.ProjectSummary]:
        raise NotImplementedError()

    def delete_project_related_resources(self, session, name: str):
        raise NotImplementedError()

    def verify_project_has_no_related_resources(self, session, name: str):
        raise NotImplementedError()

    def is_project_exists(self, session, name: str, **kwargs):
        raise NotImplementedError()

    def list_projects(
        self,
        session,
        owner: str = None,
        format_: schemas.ProjectsFormat = schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: schemas.ProjectState = None,
        names: Optional[List[str]] = None,
    ) -> schemas.ProjectsOutput:
        return self._transform_run_db_error(
            self.db.list_projects, owner, format_, labels, state
        )

    async def get_project_resources_counters(
        self,
    ) -> Tuple[
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
    ]:
        raise NotImplementedError()

    def store_project(self, session, name: str, project: schemas.Project):
        raise NotImplementedError()

    def patch_project(
        self,
        session,
        name: str,
        project: dict,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ):
        raise NotImplementedError()

    def create_project(self, session, project: schemas.Project):
        raise NotImplementedError()

    def get_project(
        self, session, name: str = None, project_id: int = None
    ) -> schemas.Project:
        raise NotImplementedError()

    def delete_project(
        self,
        session,
        name: str,
        deletion_strategy: schemas.DeletionStrategy = schemas.DeletionStrategy.default(),
    ):
        raise NotImplementedError()

    def create_feature_set(
        self, session, project, feature_set: schemas.FeatureSet, versioned=True,
    ) -> str:
        raise NotImplementedError()

    def store_feature_set(
        self,
        session,
        project,
        name,
        feature_set: schemas.FeatureSet,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
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

    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> schemas.EntitiesOutput:
        pass

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
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureSetsOutput:
        raise NotImplementedError()

    def list_feature_sets_tags(
        self, session, project: str,
    ):
        raise NotImplementedError()

    def patch_feature_set(
        self,
        session,
        project,
        name,
        feature_set_patch: dict,
        tag=None,
        uid=None,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> str:
        raise NotImplementedError()

    def delete_feature_set(self, session, project, name, tag=None, uid=None):
        raise NotImplementedError()

    def create_feature_vector(
        self, session, project, feature_vector: schemas.FeatureVector, versioned=True,
    ) -> str:
        raise NotImplementedError()

    def get_feature_vector(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> schemas.FeatureVector:
        raise NotImplementedError()

    def list_feature_vectors(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureVectorsOutput:
        raise NotImplementedError()

    def list_feature_vectors_tags(
        self, session, project: str,
    ):
        raise NotImplementedError()

    def store_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector: schemas.FeatureVector,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        raise NotImplementedError()

    def patch_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector_update: dict,
        tag=None,
        uid=None,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ) -> str:
        raise NotImplementedError()

    def delete_feature_vector(self, session, project, name, tag=None, uid=None):
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
        concurrency_limit: int,
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
        concurrency_limit: int = None,
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

    def delete_schedules(self, session, project: str):
        raise NotImplementedError()

    @staticmethod
    def _transform_run_db_error(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RunDBError as exc:
            raise DBError(exc.args)
