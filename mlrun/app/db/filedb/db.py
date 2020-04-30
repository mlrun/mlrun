from mlrun.app.db.base import DBInterface
from mlrun.db.filedb import FileRunDB


class FileDB(DBInterface):
    def __init__(self, dirpath="", format=".yaml"):
        self.db = FileRunDB(dirpath, format)

    def initialize(self, session):
        self.db.connect()

    def store_log(self, session, uid, project="", body=None, append=False):
        return self.db.store_log(uid, project, body, append)

    def get_log(self, session, uid, project="", offset=0, size=0):
        return self.db.get_log(uid, project, offset, size)

    def store_run(self, session, struct, uid, project="", iter=0):
        return self.db.store_run(struct, uid, project, iter)

    def update_run(self, session, updates: dict, uid, project="", iter=0):
        return self.db.update_run(updates, uid, project, iter)

    def read_run(self, session, uid, project="", iter=0):
        return self.db.read_run(uid, project, iter)

    def list_runs(
            self, session, name="", uid=None, project="", labels=None,
            state="", sort=True, last=0, iter=False):
        return self.db.list_runs(name, uid, project, labels, state, sort, last, iter)

    def del_run(self, session, uid, project="", iter=0):
        return self.db.del_run(uid, project, iter)

    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        return self.db.del_runs(name, project, labels, state, days_ago)

    def store_artifact(self, session, key, artifact, uid, iter=None, tag="", project=""):
        return self.db.store_artifact(key, artifact, uid, iter, tag, project)

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        return self.db.read_artifact(key, tag, iter, project)

    def list_artifacts(
            self, session, name="", project="", tag="", labels=None,
            since=None, until=None):
        return self.db.list_artifacts(name, project, tag, labels, since, until)

    def del_artifact(self, session, key, tag="", project=""):
        return self.db.del_artifact(key, tag, project)

    def del_artifacts(
            self, session, name="", project="", tag="", labels=None):
        return self.db.del_artifacts(name, project, tag, labels)

    def store_function(self, session, func, name, project="", tag=""):
        return self.db.store_function(func, name, project, tag)

    def get_function(self, session, name, project="", tag=""):
        return self.db.get_function(name, project, tag)

    def list_functions(self, session, name, project="", tag="", labels=None):
        return self.db.list_functions(name, project, tag, labels)

    def store_schedule(self, session, data):
        return self.db.store_schedule(data)

    def list_schedules(self, session):
        return self.db.list_schedules()

    def list_projects(self, session):
        return self.db.list_projects()

    def list_artifact_tags(self, session, project):
        return self.db.list_artifact_tags(project)
