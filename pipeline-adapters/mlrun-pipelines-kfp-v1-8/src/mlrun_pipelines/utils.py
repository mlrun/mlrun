# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import tempfile
import typing

from kfp_server_api import OpenApiException

import mlrun.utils
import mlrun_pipelines.common.models
import mlrun_pipelines.helpers
import mlrun_pipelines.imports


class ExtendedKfpClient(mlrun_pipelines.imports.Client):
    def retry_run(
        self,
        run_id: str,
    ) -> str:
        """
        Retries a given run by its run ID. If the run is not in a valid state for retry,
        it creates a new run with the same pipeline and parameters.

        :param run_id: The ID of the run to retry.
        :type run_id: str
        :raises ApiException: If the API request fails during the retry or new run creation process.
        :raises ValueError: If the experiment ID cannot be found for the given run ID, or if
                            the original run does not contain a valid pipeline specification.
        :raises FileNotFoundError: If a temporary file for the workflow manifest cannot be created or accessed.
        :return: The ID of the new or retried run.
        :rtype: str
        """
        # Fetch run details
        run_details = self.get_run(run_id).run
        run_status = run_details.status

        # Extract experiment ID from resource_references
        experiment_id = next(
            (
                ref.key.id
                for ref in run_details.resource_references
                if ref.key.type == "EXPERIMENT"
            ),
            None,
        )
        if not experiment_id:
            raise ValueError(f"Experiment ID not found for run ID: {run_id}")

        valid_states_for_retry = {
            mlrun_pipelines.common.models.RunStatuses.failed,
            mlrun_pipelines.common.models.RunStatuses.error,
        }
        if run_status in valid_states_for_retry:
            try:
                self._experiment_api.api_client.call_api(
                    f"/apis/v1beta1/runs/{run_id}/retry",
                    "POST",
                    response_type="ApiRun",
                    auth_settings=["BearerToken"],
                )
            except OpenApiException as error:
                mlrun.utils.logger.error(
                    "Could not trigger retry for run.", run_id=run_id, error=error
                )
                raise error
            return run_id
        else:
            # If not retryable, create a new run
            pipeline_spec = run_details.pipeline_spec

            if not pipeline_spec.pipeline_id and not pipeline_spec.workflow_manifest:
                raise ValueError(
                    "The original run does not contain a valid pipeline specification. "
                    "Please ensure the pipeline has either a pipeline ID or workflow manifest."
                )

            workflow_manifest_path = None
            if not pipeline_spec.pipeline_id:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as temp_file:
                    temp_file.write(pipeline_spec.workflow_manifest)
                    workflow_manifest_path = temp_file.name

            try:
                new_run = self.run_pipeline(
                    experiment_id=experiment_id,
                    job_name=f"Retry of {run_details.name}",
                    pipeline_id=pipeline_spec.pipeline_id,
                    params=pipeline_spec.parameters,
                    pipeline_package_path=workflow_manifest_path,
                )
                return new_run.id
            except OpenApiException as error:
                mlrun.utils.logger.error(
                    "Could not trigger new run for run.", run_id=run_id, error=error
                )
                raise error
            finally:
                if workflow_manifest_path and os.path.exists(workflow_manifest_path):
                    os.remove(workflow_manifest_path)


def compile_pipeline(
    artifact_path,
    cleanup_ttl,
    ops,
    pipeline,
    pipe_file: typing.Optional[str] = None,
    type_check: bool = False,
):
    if not pipe_file:
        pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    conf = mlrun_pipelines.helpers.new_pipe_metadata(
        artifact_path=artifact_path,
        cleanup_ttl=cleanup_ttl,
        op_transformers=ops,
    )
    mlrun_pipelines.imports.compiler.Compiler().compile(
        pipeline, pipe_file, type_check=type_check, pipeline_conf=conf
    )
    return pipe_file


def get_client(
    url: typing.Optional[str] = None, namespace: typing.Optional[str] = None
) -> ExtendedKfpClient:
    if url or namespace:
        return ExtendedKfpClient(host=url, namespace=namespace)
    return ExtendedKfpClient()
