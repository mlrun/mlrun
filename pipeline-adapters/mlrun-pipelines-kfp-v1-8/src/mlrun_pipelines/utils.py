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
    @staticmethod
    def _normalize_retry_run(
        original_name: str,
        project: str,
    ) -> str:
        job_name = original_name.strip()
        proj_prefix = f"{project}-"
        retry_prefix = "Retry of "

        proj_prefix_len = len(proj_prefix)
        retry_prefix_len = len(retry_prefix)

        if job_name.startswith(proj_prefix):
            job_name = job_name[proj_prefix_len:].strip()
        if job_name.startswith(retry_prefix):
            job_name = job_name[retry_prefix_len:].strip()

        return f"{project}-Retry of {job_name}"

    def retry_run(
        self,
        run_id: str,
        project: str,
    ) -> str:
        """
        Retries a given run by its run ID. If the run is not in a valid state for retry,
        it creates a new run with the same pipeline and parameters.

        :param run_id: The ID of the run to retry.
        :type run_id: str
        :param project: The name of the project for the run.
        :type project: str
        :raises ApiException: If the API request fails during the retry or new run creation process.
        :raises ValueError: If the experiment ID cannot be found for the given run ID, or if
                            the original run does not contain a valid pipeline specification.
        :raises FileNotFoundError: If a temporary file for the workflow manifest cannot be created or accessed.
        :return: The ID of the new or retried run.
        :rtype: str
        """
        # Fetch run details
        run_details = self.get_run(run_id).run

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

        # When retrying a KFP pipeline, we fetch the pipeline parameters from the previous run.
        # Due to an issue with the KFP server API, the pipeline parameters are returned as a list
        # containing a dictionary instead of a dictionary. We need to extract the dictionary from the list.
        pipeline_parameters = pipeline_spec.parameters
        if isinstance(pipeline_parameters, list):
            pipeline_parameters = pipeline_parameters[0]

        desired_prefix = f"{project}-Retry of "
        desired_prefix_lower = desired_prefix.lower()
        current_name = run_details.name.strip()

        if current_name.lower().startswith(desired_prefix_lower):
            job_name = current_name
        else:
            job_name = self._normalize_retry_run(current_name, project)
        try:
            new_run = self.run_pipeline(
                experiment_id=experiment_id,
                job_name=job_name,
                pipeline_id=pipeline_spec.pipeline_id,
                params=pipeline_parameters,
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
