import json
from typing import Any

from kfp_server_api.models.api_run_detail import ApiRunDetail

from mlrun.pipelines.common.helpers import (
    FlexibleMapper,
)


class PipelineManifest(FlexibleMapper):
    def __init__(self, workflow_manifest: str, pipeline_manifest: str = None):
        main_manifest = json.loads(workflow_manifest or "{}")
        if pipeline_manifest:
            pipeline_manifest = json.loads(pipeline_manifest)
            main_manifest["status"] = pipeline_manifest.get("status", {})
        super().__init__(main_manifest)


class PipelineRun(FlexibleMapper):
    _workflow_manifest: PipelineManifest

    def __init__(self, external_data: Any):
        if isinstance(external_data, ApiRunDetail):
            super().__init__(external_data.run)
            self._workflow_manifest = PipelineManifest(
                self._external_data.get("pipeline_spec", {}).get("workflow_manifest"),
                external_data.pipeline_runtime.workflow_manifest,
            )
        else:
            super().__init__(external_data)
            self._workflow_manifest = PipelineManifest(
                self._external_data.get("pipeline_spec", {}).get("workflow_manifest"),
            )

    @property
    def id(self):
        return self._external_data["id"]

    @id.setter
    def id(self, _id):
        self._external_data["id"] = _id

    @property
    def name(self):
        return self._external_data["name"]

    @name.setter
    def name(self, name):
        self._external_data["name"] = name

    @property
    def status(self):
        return self._external_data["status"]

    @status.setter
    def status(self, status):
        self._external_data["status"] = status

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
    def workflow_manifest(self) -> PipelineManifest:
        return self._workflow_manifest
