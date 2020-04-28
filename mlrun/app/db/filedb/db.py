from mlrun.app.db.base import DBInterface
from mlrun.db.filedb import FileRunDB


class FileDB(DBInterface):
    def __init__(self, dirpath="", format=".yaml"):
        self.db = FileRunDB(dirpath, format)

    def initialize(self, session):
        self.db.connect()

    def store_log(self, session, uid, project="", body=None, append=False):
        self.db.store_log(uid, project, body, append)

    def get_log(self, session, uid, project="", offset=0, size=0):
        self.db.get_log(uid, project, offset, size)

    def store_run(self, session, struct, uid, project="", iter=0):
        self.db.store_run(struct, uid, project, iter)

    def update_run(self, session, updates: dict, uid, project="", iter=0):
        self.db.update_run(updates, uid, project, iter)

    def read_run(self, session, uid, project="", iter=0):
        self.db.read_run(uid, project, iter)

    def list_runs(
            self, session, name="", uid=None, project="", labels=None,
            state="", sort=True, last=0, iter=False):
        self.db.list_runs(name, uid, project, labels, state, sort, last, iter)

    def del_run(self, session, uid, project="", iter=0):
        self.db.del_run(uid, project, iter)

    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        self.db.del_runs(name, project, labels, state, days_ago)

    def store_artifact(self, session, key, artifact, uid, iter=None, tag="", project=""):
        self.db.store_artifact(key, artifact, uid, iter, tag, project)

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        self.db.read_artifact(key, tag, iter, project)

    def list_artifacts(
            self, session, name="", project="", tag="", labels=None,
            since=None, until=None):
        self.db.list_artifacts(name, project, tag, labels, since, until)

    def del_artifact(self, session, key, tag="", project=""):
        self.db.del_artifact(key, tag, project)

    def del_artifacts(
            self, session, name="", project="", tag="", labels=None):
        self.db.del_artifacts(name, project, tag, labels)

    def store_function(self, session, func, name, project="", tag=""):
        self.db.store_function(func, name, project, tag)

    def get_function(self, session, name, project="", tag=""):
        self.db.get_function(name, project, tag)

    def list_functions(self, session, name, project="", tag="", labels=None):
        self.db.list_functions(name, project, tag, labels)

    def store_schedule(self, session, data):
        self.db.store_schedule(data)

    def list_schedules(self, session):
        self.db.list_schedules()

    def list_projects(self, session):
        self.db.list_projects()

    def list_artifact_tags(self, session, project):
        self.db.list_artifact_tags(project)
