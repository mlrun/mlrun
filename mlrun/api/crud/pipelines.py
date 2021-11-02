import ast
import http
import json
import tempfile
import traceback
import typing

import kfp
import sqlalchemy.orm

import mlrun
import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.errors
import mlrun.kfpops
import mlrun.utils.helpers
import mlrun.utils.singleton
from mlrun.utils import logger


class Pipelines(metaclass=mlrun.utils.singleton.Singleton,):
    def list_pipelines(
        self,
        db_session: sqlalchemy.orm.Session,
        project: str,
        namespace: str = mlrun.mlconf.namespace,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.metadata_only,
        page_size: typing.Optional[int] = None,
    ) -> typing.Tuple[int, typing.Optional[int], typing.List[dict]]:
        if project != "*" and (page_token or page_size or sort_by or filter_):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Filtering by project can not be used together with pagination, sorting, or custom filter"
            )
        if format_ == mlrun.api.schemas.PipelinesFormat.summary:
            # we don't support summary format in list pipelines since the returned runs doesn't include the workflow
            # manifest status that includes the nodes section we use to generate the DAG.
            # (There is a workflow manifest under the run's pipeline_spec field, but it doesn't include the status)
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Summary format is not supported for list pipelines, use get instead"
            )
        kfp_client = kfp.Client(namespace=namespace)
        if project != "*":
            run_dicts = []
            while page_token is not None:
                response = kfp_client._run_api.list_runs(
                    page_token=page_token,
                    page_size=mlrun.api.schemas.PipelinesPagination.max_page_size,
                )
                run_dicts.extend([run.to_dict() for run in response.runs or []])
                page_token = response.next_page_token
            project_runs = []
            for run_dict in run_dicts:
                run_project = self.resolve_project_from_pipeline(run_dict)
                if run_project == project:
                    project_runs.append(run_dict)
            runs = project_runs
            total_size = len(project_runs)
            next_page_token = None
        else:
            response = kfp_client._run_api.list_runs(
                page_token=page_token,
                page_size=page_size
                or mlrun.api.schemas.PipelinesPagination.default_page_size,
                sort_by=sort_by,
                filter=filter_,
            )
            runs = [run.to_dict() for run in response.runs or []]
            total_size = response.total_size
            next_page_token = response.next_page_token
        runs = self._format_runs(db_session, runs, format_)

        return total_size, next_page_token, runs

    def get_pipeline(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: typing.Optional[str] = None,
        namespace: str = mlrun.mlconf.namespace,
        format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.summary,
    ):
        kfp_client = kfp.Client(namespace=namespace)
        run = None
        try:
            api_run_detail = kfp_client.get_run(run_id)
            if api_run_detail.run:
                run = api_run_detail.to_dict()["run"]
                if project and project != "*":
                    run_project = self.resolve_project_from_pipeline(run)
                    if run_project != project:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"Pipeline run with id {run_id} is not of project {project}"
                        )
                run = self._format_run(
                    db_session, run, format_, api_run_detail.to_dict()
                )

        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting kfp run: {exc}"
            ) from exc

        return run

    def create_pipeline(
        self,
        experiment_name: str,
        run_name: str,
        content_type: str,
        data: bytes,
        arguments: dict = None,
        namespace: str = mlrun.mlconf.namespace,
    ):
        if arguments is None:
            arguments = {}
        if "/yaml" in content_type:
            content_type = ".yaml"
        elif " /zip" in content_type:
            content_type = ".zip"
        else:
            mlrun.api.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value,
                reason=f"unsupported pipeline type {content_type}",
            )

        logger.debug("Writing pipeline to temp file", content_type=content_type)
        print(str(data))

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
            kfp_client = kfp.Client(namespace=namespace)
            experiment = kfp_client.create_experiment(name=experiment_name)
            run = kfp_client.run_pipeline(
                experiment.id, run_name, pipeline_file.name, params=arguments
            )
        except Exception as exc:
            logger.warning(
                "Failed creating pipeline",
                traceback=traceback.format_exc(),
                exc=str(exc),
            )
            raise mlrun.errors.MLRunBadRequestError(f"Failed creating pipeline: {exc}")
        finally:
            pipeline_file.close()

        return run

    def _format_runs(
        self,
        db_session: sqlalchemy.orm.Session,
        runs: typing.List[dict],
        format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.metadata_only,
    ) -> typing.List[dict]:
        formatted_runs = []
        for run in runs:
            formatted_runs.append(self._format_run(db_session, run, format_))
        return formatted_runs

    def _format_run(
        self,
        db_session: sqlalchemy.orm.Session,
        run: dict,
        format_: mlrun.api.schemas.PipelinesFormat = mlrun.api.schemas.PipelinesFormat.metadata_only,
        api_run_detail: typing.Optional[dict] = None,
    ) -> dict:
        run["project"] = self.resolve_project_from_pipeline(run)
        if format_ == mlrun.api.schemas.PipelinesFormat.full:
            return run
        elif format_ == mlrun.api.schemas.PipelinesFormat.metadata_only:
            return {
                k: str(v)
                for k, v in run.items()
                if k
                in [
                    "id",
                    "name",
                    "project",
                    "status",
                    "error",
                    "created_at",
                    "scheduled_at",
                    "finished_at",
                    "description",
                ]
            }
        elif format_ == mlrun.api.schemas.PipelinesFormat.name_only:
            return run.get("name")
        elif format_ == mlrun.api.schemas.PipelinesFormat.summary:
            if not api_run_detail:
                raise mlrun.errors.MLRunRuntimeError(
                    "The full kfp api_run_detail object is needed to generate the summary format"
                )
            return mlrun.kfpops.format_summary_from_kfp_run(
                api_run_detail, run["project"], db_session
            )
        else:
            raise NotImplementedError(
                f"Provided format is not supported. format={format_}"
            )

    def _resolve_project_from_command(
        self,
        command: typing.List[str],
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
                        ) = mlrun.utils.helpers.parse_versioned_object_uri(
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
                            "Failed parsing runtime. Skipping", runtime=runtime, exc=exc
                        )
                    else:
                        if isinstance(parsed_runtime, dict):
                            project = parsed_runtime.get("metadata", {}).get("project")
                            if project:
                                return project

        return None

    def resolve_project_from_pipeline(self, pipeline):

        workflow_manifest = json.loads(
            pipeline.get("pipeline_spec", {}).get("workflow_manifest") or "{}"
        )
        return self.resolve_project_from_workflow_manifest(workflow_manifest)

    def resolve_project_from_workflow_manifest(self, workflow_manifest):
        templates = workflow_manifest.get("spec", {}).get("templates", [])
        for template in templates:
            project_from_annotation = (
                template.get("metadata", {})
                .get("annotations", {})
                .get(mlrun.kfpops.project_annotation)
            )
            if project_from_annotation:
                return project_from_annotation
            command = template.get("container", {}).get("command", [])
            action = None
            for index, argument in enumerate(command):
                if argument == "mlrun" and index + 1 < len(command):
                    action = command[index + 1]
                    break
            if action:
                if action == "deploy":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=True,
                        has_func_url_flags=True,
                        has_runtime_flags=False,
                    )
                    if project:
                        return project
                elif action == "run":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=False,
                        has_func_url_flags=True,
                        has_runtime_flags=True,
                    )
                    if project:
                        return project
                elif action == "build":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=False,
                        has_func_url_flags=False,
                        has_runtime_flags=True,
                    )
                    if project:
                        return project
                else:
                    raise NotImplementedError(f"Unknown action: {action}")

        return mlrun.mlconf.default_project
