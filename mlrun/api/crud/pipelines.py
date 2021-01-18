import ast
import json
import typing

import kfp

import mlrun
import mlrun.api.schemas
import mlrun.errors
import mlrun.utils.helpers
from mlrun.utils import logger


# TODO: changed to be under a singleton like projects and runtimes
def list_pipelines(
    project: str,
    namespace: str = "",
    sort_by: str = "",
    page_token: str = "",
    filter_: str = "",
    format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.metadata_only,
    page_size: typing.Optional[int] = None,
) -> typing.Tuple[int, typing.Optional[int], typing.List[dict]]:
    if project != "*" and (page_token or page_size or sort_by or filter_):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Filtering by project can not be used together with pagination, sorting, or custom filter"
        )
    namespace = namespace or mlrun.mlconf.namespace
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
            run_project = _resolve_pipeline_project(run_dict)
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
    runs = _format_runs(runs, format_)

    return total_size, next_page_token, runs


def _format_runs(
    runs: typing.List[dict],
    format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.metadata_only,
) -> typing.List[dict]:
    if format_ == mlrun.api.schemas.Format.full:
        return runs
    elif format_ == mlrun.api.schemas.Format.metadata_only:
        formatted_runs = []
        for run in runs:
            formatted_runs.append(
                {
                    k: str(v)
                    for k, v in run.items()
                    if k
                    in [
                        "id",
                        "name",
                        "status",
                        "error",
                        "created_at",
                        "scheduled_at",
                        "finished_at",
                        "description",
                    ]
                }
            )
        return formatted_runs
    elif format_ == mlrun.api.schemas.Format.name_only:
        formatted_runs = []
        for run in runs:
            formatted_runs.append(run.get("name"))
        return formatted_runs
    else:
        raise NotImplementedError(f"Provided format is not supported. format={format_}")


def _resolve_project_from_command(
    command: typing.List[str],
    hyphen_p_is_also_project: bool,
    has_func_url_flags: bool,
    has_runtime_flags: bool,
):
    # project has precedence over function url so search for it first
    for index, argument in enumerate(command):
        if (
            (argument == "-p" and hyphen_p_is_also_project) or argument == "--project"
        ) and index + 1 < len(command):
            return command[index + 1]
    if has_func_url_flags:
        for index, argument in enumerate(command):
            if (argument == "-f" or argument == "--func-url") and index + 1 < len(
                command
            ):
                function_url = command[index + 1]
                if function_url.startswith("db://"):
                    project, _, _, _ = mlrun.utils.helpers.parse_versioned_object_uri(
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


def _resolve_pipeline_project(pipeline):
    workflow_manifest = json.loads(
        pipeline.get("pipeline_spec", {}).get("workflow_manifest", "{}")
    )
    templates = workflow_manifest.get("spec", {}).get("templates", [])
    for template in templates:
        command = template.get("container", {}).get("command", [])
        action = None
        for index, argument in enumerate(command):
            if argument == "mlrun" and index + 1 < len(command):
                action = command[index + 1]
                break
        if action:
            if action == "deploy":
                project = _resolve_project_from_command(
                    command,
                    hyphen_p_is_also_project=True,
                    has_func_url_flags=True,
                    has_runtime_flags=False,
                )
                if project:
                    return project
            elif action == "run":
                project = _resolve_project_from_command(
                    command,
                    hyphen_p_is_also_project=False,
                    has_func_url_flags=True,
                    has_runtime_flags=True,
                )
                if project:
                    return project
            elif action == "build":
                project = _resolve_project_from_command(
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
