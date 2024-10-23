# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import typing
from time import sleep

import mlrun_pipelines
import pytz
from dateutil import parser
from kfp import Client
from mlrun_pipelines.models import PipelineRun

import mlrun


def validate_and_convert_date(date_input: str) -> str:
    """
    Converts any recognizable date string into a standardized RFC 3339 format.
    :param date_input: A date string in a recognizable format.
    """
    try:
        # Parse the date using dateutil.parser
        dt_object = parser.parse(date_input)

        # Check if the parsed date has timezone information
        if dt_object.tzinfo is not None:
            # Convert to UTC if it's in a different timezone
            dt_object = dt_object.astimezone(pytz.utc)
        else:
            # If no timezone info is present, assume it's in local time
            local_tz = pytz.timezone("UTC")
            dt_object = local_tz.localize(dt_object)

        formatted_date = dt_object.isoformat().replace("+00:00", "Z")
        if not formatted_date.endswith("Z"):
            formatted_date += "Z"

        # Return the date in RFC 3339 format with 'Z' for UTC
        return formatted_date
    except (ValueError, OverflowError) as e:
        raise ValueError(
            f"Invalid date format: {date_input}. Please provide a valid date."
        ) from e


def get_kfp_client(
    kfp_url=mlrun.mlconf.kfp_url, namespace: str = "default-tenant"
) -> Client:
    kfp_client = mlrun_pipelines.utils.get_client(kfp_url, namespace)
    return kfp_client


def get_experiment_name(kfp_client: Client, experiment_id: str) -> str:
    experiment = kfp_client.get_experiment(experiment_id=experiment_id)
    return experiment.name if experiment else ""


def filter_out_non_related_runs(
    project_name: str, runs: list[PipelineRun]
) -> list[PipelineRun]:
    project_runs = []
    for run in runs:
        run_project = mlrun_pipelines.mixins.PipelineProviderMixin().resolve_project_from_workflow_manifest(
            run.workflow_manifest()
        )
        if run_project == project_name:
            project_runs.append(run)
    return project_runs


def get_list_runs_filter(project_name: str, end_date: str, start_date: str) -> str:
    filters = {
        "predicates": [
            {
                "key": "created_at",
                "op": 7,  # Operation 7 corresponds to '<=' (less than or equal)
                "timestamp_value": end_date,
            },
        ]
    }
    if not project_name == "*":
        filters["predicates"].append(
            {
                "key": "name",
                "op": 9,  # Operation 9 corresponds to substring matching
                "string_value": project_name,
            }
        )
    if start_date:
        filters["predicates"].append(
            {
                "key": "created_at",
                "op": 5,  # Operation 5 corresponds to '>=' (greater than or equal)
                "timestamp_value": start_date,
            }
        )
    return json.dumps(filters)


def list_pipelines_runs(
    kfp_client: Client, query_filter: str, page_token: str = "", sort_by: str = ""
) -> list[PipelineRun]:
    runs = []
    counter = 0
    while page_token is not None:
        # kfp doesn't allow us to pass both a page_token and the `filter` and `sort_by` params.
        # When we have a token from previous call, we will strip out the filter and use the token to continue
        # (the token contains the details of the filter that was used to create it)
        response = kfp_client.list_runs(
            page_token=page_token,
            page_size=mlrun.common.schemas.PipelinesPagination.max_page_size,
            sort_by=sort_by if page_token == "" else "",
            filter=query_filter if page_token == "" else "",
        )
        runs.extend([PipelineRun(run) for run in response.runs or []])
        page_token = response.next_page_token
        if counter % 50 == 0:
            mlrun.utils.logger.info(
                "Collecting pipelines runs:", runs_count=(10000 * counter)
            )
        counter += 1
    return runs


def query_and_filter_runs(
    kfp_client: Client, project_name: str, query_filter: str
) -> tuple[list[tuple[str, str]], set]:
    """
    Query the pipeline runs and filter them based on the project name.

    :param kfp_client: KFP client for interacting with the pipeline API.
    :param project_name: Name of the project for filtering the runs.
    :param query_filter: Filter for querying the runs.
    """
    runs = list_pipelines_runs(kfp_client, query_filter)

    # Filter out non project-related runs if project was provided
    project_runs = (
        filter_out_non_related_runs(project_name, runs) if project_name != "*" else runs
    )
    if project_name == "*":
        project_names = [
            mlrun_pipelines.mixins.PipelineProviderMixin().resolve_project_from_workflow_manifest(
                run.workflow_manifest()
            )
            for run in project_runs
        ]
        project_names = set(project_names)
    else:
        project_names = {project_name}

    mlrun.utils.logger.info(
        f"Found {len(project_runs)} runs for projects", project_names=project_names
    )
    runs_ids = [(run.id, run.name) for run in project_runs]

    # Collect experiment IDs
    experiment_ids = set(run.experiment_id for run in project_runs if run.experiment_id)

    return runs_ids, experiment_ids


def find_empty_experiments(
    kfp_client: Client, experiments_ids: set
) -> list[tuple[str, str]]:
    # Find empty experiments
    empty_experiment_ids = []
    for experiment_id in experiments_ids:
        runs = kfp_client.list_runs(experiment_id=experiment_id)

        if not runs.total_size:
            experiment_name = get_experiment_name(kfp_client, experiment_id)
            empty_experiment_ids.append((experiment_id, experiment_name))
    return empty_experiment_ids


def delete_runs(
    context: mlrun.MLClientCtx,
    kfp_client: Client,
    runs_ids: list[tuple[str, str]],
    dry_run: bool,
) -> None:
    """
    Delete pipeline runs based on the provided run IDs.

    :param context: The context object to log results.
    :param runs_ids: List of tuples containing run IDs and names.
    :param kfp_client: The KFP client used to interact with the pipeline API.
    :param dry_run: If True, perform a dry run (only log what would be deleted).
    """
    delete_items(
        context,
        runs_ids,
        lambda run_id: kfp_client._run_api.delete_run(run_id),
        item_type="run",
        dry=dry_run,
    )


def delete_empty_experiments(
    context: mlrun.MLClientCtx,
    kfp_client: Client,
    experiments_ids: set[str],
    dry_run: bool,
) -> None:
    """
    Find and delete empty experiments based on the provided experiment IDs.

    :param context: The context object to log results.
    :param kfp_client: The KFP client used to interact with the pipeline API.
    :param experiments_ids: List of experiment IDs to check for emptiness.
    :param dry_run: If True, perform a dry run (only log what would be deleted).
    """
    empty_experiment_ids = find_empty_experiments(kfp_client, experiments_ids)

    delete_items(
        context,
        empty_experiment_ids,
        lambda experiment_id: kfp_client._experiment_api.delete_experiment(
            id=experiment_id
        ),
        item_type="experiment",
        dry=dry_run,
    )


def delete_items(
    context: mlrun.MLClientCtx,
    items: list[tuple[str, str]],
    delete_func: typing.Callable[[str], None],
    item_type: str = "run",
    log_key_total: typing.Optional[str] = None,
    log_key_succeeded: typing.Optional[str] = None,
    log_key_failed: typing.Optional[str] = None,
    dry: bool = True,
) -> None:
    """
    A generic function to delete items such as runs or experiments and log the results.

    :param context: The context object to log results.
    :param item_ids: A list of item IDs to be deleted.
    :param delete_func: The function responsible for deleting each item.
                        It should take an ID as its argument.
    :param item_type: The type of items being deleted (used for logging).
                      Defaults to "run".
    :param log_key_total: The key for logging the total number of items.
                          Defaults to "runs_total" for runs.
    :param log_key_succeeded: The key for logging the number of successfully deleted items.
                              Defaults to "runs_num_of_succeeded" for runs.
    :param log_key_failed: The key for logging the number of failed deletions.
                           Defaults to "runs_num_of_failed" for runs.
    :param dry: If True, perform a dry run (only log what would be deleted).
    """
    total = len(items)
    num_of_succeeded = 0
    num_of_failed = 0
    failed_items = []
    succeeded_items = []

    context.log_result(key=log_key_total or f"{item_type}s_total", value=total)
    mlrun.utils.logger.info(f"Starting to delete {total} {item_type}s")

    if not dry:
        for run_id, name in items:
            try:
                delete_func(run_id)
                num_of_succeeded += 1
                succeeded_items.append((name, run_id))
                if num_of_succeeded % 100 == 0:
                    mlrun.utils.logger.info(
                        f"Deleted {num_of_succeeded}/{total} {item_type}s successfully"
                    )
                    sleep(10)
            except Exception as exc:
                num_of_failed += 1
                failed_items.append((name, run_id, exc, exc.reason))
                mlrun.utils.logger.warning(
                    f"Failed to delete {item_type} '{name}' with ID: {run_id}. Error: {exc}"
                )

        # Log results
        context.log_result(
            key=log_key_succeeded or f"{item_type}s_num_of_succeeded",
            value=num_of_succeeded,
        )
        context.log_result(
            key=f"{item_type}s_succeeded",
            value=succeeded_items,
        )
        context.log_result(
            key=log_key_failed or f"{item_type}s_num_of_failed", value=num_of_failed
        )
        context.log_result(key=f"{item_type}s_failed", value=failed_items)

    else:
        mlrun.utils.logger.info(
            f"Dry run: The following {item_type}s would be deleted: {items}"
        )
        context.log_result(key=f"{item_type}s_to_be_deleted", value=items)


def delete_project_old_pipelines(
    context: mlrun.MLClientCtx,
    project_name: str,
    end_date: str,
    start_date: str = "",
    dry_run: bool = True,
) -> None:
    """
    Delete old pipeline runs associated with a specific project.

    This function retrieves all pipeline runs for the given project, filters them based on the
    provided date range, and deletes both the runs .

    :param context: The context object to log results.
    :param project_name: Name of the project for which to delete old pipelines.
    :param end_date: The cutoff date for deleting pipeline runs. All runs created on or before
                     this date will be considered for deletion.
    :param start_date: (Optional) The start date for filtering pipeline runs. If provided, only
                       runs created on or after this date will be considered for deletion.
                       Defaults to an empty string, which means no start date filtering.
    :param dry_run: If True, perform a dry run (only log what would be deleted).

    """
    # Validate and convert dates
    end_date = validate_and_convert_date(end_date)
    start_date = "" if not start_date else validate_and_convert_date(start_date)

    # get KFP client
    kfp_client = get_kfp_client()

    # Generate filter and query runs
    query_filter = get_list_runs_filter(project_name, end_date, start_date)

    # Query and filter runs
    runs_ids, experiments_ids = query_and_filter_runs(
        kfp_client, project_name, query_filter
    )

    # Delete runs
    delete_runs(context, kfp_client, runs_ids, dry_run)

    # Find and delete empty experiments
    delete_empty_experiments(context, kfp_client, experiments_ids, dry_run)
