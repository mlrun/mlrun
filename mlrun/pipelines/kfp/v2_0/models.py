from mlrun.pipelines.common.helpers import FlexibleMapper


class PipelineRun(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["run_id"]

    @id.setter
    def id(self, _id):
        self._external_data["run_id"] = _id

    @property
    def name(self):
        return self._external_data["display_name"]

    @name.setter
    def name(self, name):
        self._external_data["display_name"] = name

    @property
    def status(self):
        return self._external_data["state"]

    @status.setter
    def status(self, status):
        self._external_data["state"] = status

    @property
    def description(self):
        return self._external_data["description"]

    @description.setter
    def description(self, description):
        self._external_data["description"] = description

    @property
    def created_at(self):
        return self._external_data["created_at"]

    @created_at.setter
    def created_at(self, created_at):
        self._external_data["created_at"] = created_at

    @property
    def scheduled_at(self):
        return self._external_data["scheduled_at"]

    @scheduled_at.setter
    def scheduled_at(self, scheduled_at):
        self._external_data["scheduled_at"] = scheduled_at

    @property
    def finished_at(self):
        return self._external_data["finished_at"]

    @finished_at.setter
    def finished_at(self, finished_at):
        self._external_data["finished_at"] = finished_at

    @property
    def workflow_manifest(self) -> dict:
        return self._external_data["pipeline_spec"]
