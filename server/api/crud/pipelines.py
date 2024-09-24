# Copyright 2023 Iguazio
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
import ast
import http
import tempfile
import traceback
import typing

import kfp_server_api
import mlrun_pipelines.utils
import sqlalchemy.orm
from mlrun_pipelines.mixins import PipelineProviderMixin
from mlrun_pipelines.models import PipelineExperiment, PipelineRun

import mlrun
import mlrun.common.formatters
import mlrun.common.helpers
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.helpers
import mlrun.utils.singleton
import server.api.api.utils
from mlrun.errors import err_to_str
from mlrun.utils import logger


class Pipelines(
    PipelineProviderMixin,
    metaclass=mlrun.utils.singleton.Singleton,
):
    def list_pipelines(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        namespace: typing.Optional[str] = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        name_contains: str = "",
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        page_size: typing.Optional[int] = None,
    ) -> tuple[int, typing.Optional[int], list[dict]]:
        if project != "*" and (page_token or page_size):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Filtering by project can not be used together with pagination"
            )
        if format_ == mlrun.common.formatters.PipelineFormat.summary:
            # we don't support summary format in list pipelines since the returned runs doesn't include the workflow
            # manifest status that includes the nodes section we use to generate the DAG.
            # (There is a workflow manifest under the run's pipeline_spec field, but it doesn't include the status)
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Summary format is not supported for list pipelines, use get instead"
            )

        kfp_client = self.initialize_kfp_client(namespace)
        if project != "*":
            # If no filter is provided and the project is not "*",
            # automatically apply a filter to match runs where the project name
            # is a substring of the pipeline's name. This ensures that only pipelines
            # with the project name in their name are returned, helping narrow down the results.
            if not filter_:
                logger.debug(
                    "No filter provided. "
                    "Applying project-based filter for project to match pipelines with project name as a substring",
                    project=project,
                )
                filter_ = mlrun.utils.get_kfp_project_filter(project_name=project)
            runs = []
            while page_token is not None:
                # kfp doesn't allow us to pass both a page_token and the `filter` and `sort_by` params.
                # When we have a token from previous call, we will strip out the filter and use the token to continue
                # (the token contains the details of the filter that was used to create it)
                response = kfp_client.list_runs(
                    page_token=page_token,
                    page_size=mlrun.common.schemas.PipelinesPagination.max_page_size,
                    sort_by=sort_by if page_token == "" else "",
                    filter=filter_ if page_token == "" else "",
                )
                runs.extend([PipelineRun(run) for run in response.runs or []])
                page_token = response.next_page_token
            project_runs = []
            for run in runs:
                run_project = self.resolve_project_from_pipeline(run)
                if run_project == project:
                    project_runs.append(run)
            runs = self._filter_runs_by_name(project_runs, name_contains)

            total_size = len(runs)
            next_page_token = None
        else:
            try:
                response = kfp_client.list_runs(
                    page_token=page_token,
                    page_size=page_size
                    or mlrun.common.schemas.PipelinesPagination.default_page_size,
                    sort_by=sort_by,
                    filter=filter_,
                )
            except kfp_server_api.ApiException as exc:
                # extract the summary of the error message from the exception
                error_message = exc.body or exc.reason or exc
                if "message" in error_message:
                    error_message = error_message["message"]
                raise mlrun.errors.err_for_status_code(
                    exc.status, err_to_str(error_message)
                ) from exc
            runs = [PipelineRun(run) for run in response.runs or []]
            runs = self._filter_runs_by_name(runs, name_contains)
            next_page_token = response.next_page_token
            # In-memory filtering turns Kubeflow's counting inaccurate if there are multiple pages of data
            # so don't pass it to the client in such case
            if next_page_token:
                total_size = -1
            else:
                total_size = len(runs)
        runs = self._format_runs(runs, format_, kfp_client)

        return total_size, next_page_token, runs

    def delete_pipelines_runs(
        self, db_session: sqlalchemy.orm.Session, project_name: str
    ):
        _, _, project_pipeline_runs = self.list_pipelines(
            db_session=db_session,
            project=project_name,
            format_=mlrun.common.formatters.PipelineFormat.metadata_only,
        )
        kfp_client = self.initialize_kfp_client()

        if project_pipeline_runs:
            logger.debug(
                "Detected pipeline runs for project, deleting them",
                project_name=project_name,
                pipeline_run_ids=[run["id"] for run in project_pipeline_runs],
            )

        succeeded = 0
        failed = 0
        experiment_ids = set()
        for pipeline_run in project_pipeline_runs:
            try:
                pipeline_run = PipelineRun(pipeline_run)
                # delete pipeline run also terminates it if it is in progress
                kfp_client._run_api.delete_run(pipeline_run.id)
                if pipeline_run.experiment_id:
                    experiment_ids.add(pipeline_run.experiment_id)
                succeeded += 1
            except Exception as exc:
                # we don't want to fail the entire delete operation if we failed to delete a single pipeline run
                # so it won't fail the delete project operation. we will log the error and continue
                logger.warning(
                    "Failed to delete pipeline run",
                    project_name=project_name,
                    pipeline_run_id=pipeline_run.id,
                    exc_info=exc,
                )
                failed += 1
        logger.debug(
            "Finished deleting pipeline runs",
            project_name=project_name,
            succeeded=succeeded,
            failed=failed,
        )

        succeeded = 0
        failed = 0
        for experiment_id in experiment_ids:
            try:
                logger.debug(
                    f"Detected experiment for project {project_name} and deleting it",
                    project_name=project_name,
                    experiment_id=experiment_id,
                )
                kfp_client._experiment_api.delete_experiment(id=experiment_id)
                succeeded += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Failed to delete an experiment",
                    project_name=project_name,
                    experiment_id=experiment_id,
                    exc_info=err_to_str(exc),
                )
        logger.debug(
            "Finished deleting project experiments",
            project_name=project_name,
            succeeded=succeeded,
            failed=failed,
        )

    def get_pipeline(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: typing.Optional[str] = None,
        namespace: typing.Optional[str] = None,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.summary,
    ):
        kfp_client = self.initialize_kfp_client(namespace)
        run = None
        try:
            api_run_detail = kfp_client.get_run(run_id)
            run = PipelineRun(api_run_detail)
            if run:
                if project and project != "*":
                    run_project = self.resolve_project_from_pipeline(run)
                    if run_project != project:
                        raise mlrun.errors.MLRunNotFoundError(
                            f"Pipeline run with id {run_id} is not of project {project}"
                        )

                logger.debug(
                    "Got kfp run",
                    run_id=run_id,
                    run_name=run.get("name"),
                    project=project,
                    format_=format_,
                )
                run = self._format_run(run, format_, kfp_client)
        except kfp_server_api.ApiException as exc:
            raise mlrun.errors.err_for_status_code(exc.status, err_to_str(exc)) from exc
        except mlrun.errors.MLRunHTTPStatusError:
            raise
        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting kfp run: {err_to_str(exc)}"
            ) from exc

        return run

    def create_pipeline(
        self,
        experiment_name: str,
        run_name: str,
        content_type: str,
        data: bytes,
        arguments: dict = None,
    ):
        if arguments is None:
            arguments = {}
        if "/yaml" in content_type:
            content_type = ".yaml"
        elif " /zip" in content_type:
            content_type = ".zip"
        else:
            server.api.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value,
                reason=f"unsupported pipeline type {content_type}",
            )

        logger.debug("Writing pipeline to temp file", content_type=content_type)

        pipeline_file = tempfile.NamedTemporaryFile(suffix=content_type)
        with open(pipeline_file.name, "wb") as fp:
            fp.write(data)

        logger.info(
            "Creating pipeline",
            experiment_name=experiment_name,
            run_name=run_name,
            arguments=arguments,
        )

        try:
            kfp_client = self.initialize_kfp_client()
            experiment = PipelineExperiment(
                kfp_client.create_experiment(name=experiment_name)
            )
            run = PipelineRun(
                kfp_client.run_pipeline(
                    experiment.id, run_name, pipeline_file.name, params=arguments
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed creating pipeline",
                traceback=traceback.format_exc(),
                exc=err_to_str(exc),
            )
            raise mlrun.errors.MLRunBadRequestError(
                f"Failed creating pipeline: {err_to_str(exc)}"
            )
        finally:
            pipeline_file.close()

        return run

    @staticmethod
    def initialize_kfp_client(namespace: typing.Optional[str] = None):
        return mlrun_pipelines.utils.get_client(mlrun.mlconf.kfp_url, namespace)

    def _format_runs(
        self,
        runs: list[dict],
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        kfp_client: mlrun_pipelines.utils.kfp.Client = None,
    ) -> list[dict]:
        formatted_runs = []
        for run in runs:
            formatted_runs.append(self._format_run(run, format_, kfp_client))
        return formatted_runs

    def _format_run(
        self,
        run: PipelineRun,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        kfp_client: mlrun_pipelines.utils.kfp.Client = None,
    ) -> dict:
        run.project = self.resolve_project_from_pipeline(run)
        if kfp_client:
            error = self.get_error_from_pipeline(kfp_client, run)
            if error:
                run.error = error
        return mlrun.common.formatters.PipelineFormat.format_obj(run, format_)

    def _resolve_project_from_command(
        self,
        command: list[str],
        hyphen_p_is_also_project: bool,
        has_func_url_flags: bool,
        has_runtime_flags: bool,
    ):
        # project has precedence over function url so search for it first
        for index, argument in enumerate(command):
            if (
                (argument == "-p" and hyphen_p_is_also_project)
                or argument == "--project"
            ) and index + 1 < len(command):
                return command[index + 1]
        if has_func_url_flags:
            for index, argument in enumerate(command):
                if (argument == "-f" or argument == "--func-url") and index + 1 < len(
                    command
                ):
                    function_url = command[index + 1]
                    if function_url.startswith("db://"):
                        (
                            project,
                            _,
                            _,
                            _,
                        ) = mlrun.common.helpers.parse_versioned_object_uri(
                            function_url[len("db://") :]
                        )
                        if project:
                            return project
        if has_runtime_flags:
            for index, argument in enumerate(command):
                if (argument == "-r" or argument == "--runtime") and index + 1 < len(
                    command
                ):
                    runtime = command[index + 1]
                    try:
                        parsed_runtime = ast.literal_eval(runtime)
                    except Exception as exc:
                        logger.warning(
                            "Failed parsing runtime. Skipping",
                            runtime=runtime,
                            exc=err_to_str(exc),
                        )
                    else:
                        if isinstance(parsed_runtime, dict):
                            project = parsed_runtime.get("metadata", {}).get("project")
                            if project:
                                return project

        return None

    def resolve_project_from_pipeline(self, pipeline: PipelineRun):
        return self.resolve_project_from_workflow_manifest(pipeline.workflow_manifest())

    def get_error_from_pipeline(self, kfp_client, run: PipelineRun):
        pipeline = kfp_client.get_run(run.id)
        return self.resolve_error_from_pipeline(pipeline)

    def _filter_runs_by_name(self, runs: list, target_name: str) -> list:
        """Filter runs by their name while ignoring the project string on them

        :param runs: list of runs to be filtered
        :param target_name: string that should be part of a valid run name
        :return: filtered list of runs
        """
        if not target_name:
            return runs

        def filter_by(run):
            run_name = run.get("name", "").removeprefix(
                self.resolve_project_from_pipeline(run) + "-"
            )
            if target_name in run_name:
                return True
            return False

        return list(filter(filter_by, runs))
