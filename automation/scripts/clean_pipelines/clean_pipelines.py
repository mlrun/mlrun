# Copyright 2023 Iguazio
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

import typing
from time import sleep

import pandas as pd
from kfp import Client

import mlrun
import mlrun.utils
import mlrun_pipelines
from mlrun_pipelines.models import PipelineRun

# Interval for logging deletion progress
DELETION_LOG_INTERVAL = 100


def delete_project_old_pipelines(
    context: mlrun.MLClientCtx,
    project_name: str,
    end_date: str,
    start_date: str = "",
    dry_run: bool = False,
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
    end_date = mlrun.utils.validate_and_convert_date(end_date)
    start_date = (
        "" if not start_date else mlrun.utils.validate_and_convert_date(start_date)
    )

    # get KFP client
    kfp_client = _get_kfp_client()

    # Generate filter and query runs
    query_filter = mlrun.utils.get_kfp_list_runs_filter(
        project_name, end_date, start_date
    )

    # Query and filter runs
    runs, experiments_ids = _query_and_filter_runs(
        kfp_client, project_name, query_filter
    )

    # Delete runs
    _delete_runs_and_empty_experiments(
        context, kfp_client, runs, experiments_ids, dry_run
    )
    _delete_runs(context, kfp_client, runs, dry_run)

    # Find and delete empty experiments
    _delete_empty_experiments(context, kfp_client, experiments_ids)


def _get_kfp_client(
    kfp_url=mlrun.mlconf.kfp_url, namespace: str = mlrun.mlconf.namespace
) -> Client:
    kfp_client = mlrun_pipelines.utils.get_client(
        mlrun.utils.logger, kfp_url, namespace
    )
    return kfp_client


def _query_and_filter_runs(
    kfp_client: Client, project_name: str, query_filter: str
) -> tuple[list[tuple[str, str]], set]:
    """
    Query the pipeline runs and filter them based on the project name.

    :param kfp_client: KFP client for interacting with the pipeline API.
    :param project_name: Name of the project for filtering the runs.
    :param query_filter: Filter for querying the runs.
    """
    runs = _list_pipelines_runs(kfp_client, query_filter)

    # Filter out non-project-related runs if project was provided
    project_runs = _filter_project_runs(project_name, runs)

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
    runs = [(run.id, run.name) for run in project_runs]

    # Collect experiment IDs
    experiment_ids = set(run.experiment_id for run in project_runs if run.experiment_id)

    return runs, experiment_ids


def _list_pipelines_runs(
    kfp_client: Client,
    query_filter: str,
    page_token: str = "",
    sort_by: str = "",
    batch_size: int = 1000,
) -> list[PipelineRun]:
    runs = []
    while page_token:
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

        if len(runs) % batch_size == 0:
            mlrun.utils.logger.info(f"Collected {len(runs)} pipeline runs so far.")
    return runs


def _filter_project_runs(
    project_name: str, runs: list[PipelineRun]
) -> list[PipelineRun]:
    # If project_name is "*", return all runs without filtering
    if project_name == "*":
        return runs

    project_runs = []
    for run in runs:
        run_project = mlrun_pipelines.mixins.PipelineProviderMixin().resolve_project_from_workflow_manifest(
            run.workflow_manifest()
        )
        if run_project == project_name:
            project_runs.append(run)
    return project_runs


def _delete_runs_and_empty_experiments(
    context: mlrun.MLClientCtx,
    kfp_client: Client,
    runs: list[tuple[str, str]],
    experiments_ids: set[str],
    dry_run: bool,
) -> None:
    """
    Deletes specified pipeline runs and their corresponding empty experiments.

    This function either performs an actual deletion or a dry run depending on the `dry_run` parameter.
    If `dry_run` is True, it logs the runs that would be deleted without performing any deletion.
    If `dry_run` is False, it deletes the provided runs and then deletes any experiments
    that are left empty as a result.

    :param context: The context object to log results.
    :param kfp_client: The KFP client used to interact with the pipeline API.
    :param runs: A list of tuples representing the runs to delete, where each tuple contains (run_id, run_name).
    :param experiments_ids: A set of experiment IDs to check for emptiness after run deletion.
    :param dry_run: If True, perform a dry run by logging what would be deleted without actually deleting anything.
    """
    if not dry_run:
        # Delete runs
        _delete_runs(context, kfp_client, runs)

        # Find and delete empty experiments
        _delete_empty_experiments(context, kfp_client, experiments_ids)

    else:
        mlrun.utils.logger.info(f"Dry run: {len(runs)} runs would be deleted")
        context.log_result(key="runs_to_be_deleted", value=runs)


def _delete_runs(
    context: mlrun.MLClientCtx,
    kfp_client: Client,
    runs: list[tuple[str, str]],
) -> None:
    """
    Delete pipeline runs based on the provided runs.

    :param context: The context object to log results.
    :param runs: List of tuples containing run IDs and names.
    :param kfp_client: The KFP client used to interact with the pipeline API.
    """
    _delete_items(
        context,
        runs,
        lambda run_id: kfp_client._run_api.delete_run(run_id),
    )


def _delete_empty_experiments(
    context: mlrun.MLClientCtx,
    kfp_client: Client,
    experiments_ids: set[str],
) -> None:
    """
    Find and delete empty experiments based on the provided experiment IDs.

    :param context: The context object to log results.
    :param kfp_client: The KFP client used to interact with the pipeline API.
    :param experiments_ids: Set of experiment IDs to check for emptiness.
    """
    empty_experiment_ids = _find_empty_experiments(kfp_client, experiments_ids)

    _delete_items(
        context,
        empty_experiment_ids,
        lambda experiment_id: kfp_client._experiment_api.delete_experiment(
            id=experiment_id
        ),
        item_type="experiment",
    )


def _find_empty_experiments(
    kfp_client: Client, experiments_ids: set
) -> list[tuple[str, str]]:
    # Find empty experiments
    empty_experiment_ids = []
    for experiment_id in experiments_ids:
        runs = kfp_client.list_runs(experiment_id=experiment_id)

        if not runs.total_size:
            experiment_name = _get_experiment_name(kfp_client, experiment_id)
            empty_experiment_ids.append((experiment_id, experiment_name))
    return empty_experiment_ids


def _get_experiment_name(kfp_client: Client, experiment_id: str) -> str:
    experiment = kfp_client.get_experiment(experiment_id=experiment_id)
    return experiment.name if experiment else ""


def _delete_items(
    context: mlrun.MLClientCtx,
    items: list[tuple[str, str]],
    delete_func: typing.Callable[[str], None],
    item_type: str = "run",
) -> None:
    """
    A generic function to delete items such as runs or experiments and log the results.

    :param context: The context object to log results.
    :param items: A list of tuples, where each tuple contains the item ID and name to be deleted.
    :param delete_func: The function responsible for deleting each item.
                        It should take an ID as its argument.
    :param item_type: The type of items being deleted (used for logging).
                      Defaults to "run".
    """
    total = len(items)

    context.log_result(key=f"{item_type}s_total", value=total)
    mlrun.utils.logger.info(f"Deleting {total} {item_type}s")

    deleted, failed = _perform_deletion(items, delete_func, total, item_type)
    _log_results(context, deleted, failed, item_type)


def _perform_deletion(
    items: list[tuple[str, str]],
    delete_func: typing.Callable[[str], None],
    total_items_amount: int,
    item_type: str = "run",
) -> tuple[list[tuple[str, str]], list[tuple[str, str, Exception, str]]]:
    """
    This function iterates through a list of items, attempts to delete each item using
    the provided `delete_func`, and logs progress. It returns a list of successfully deleted
    items and a list of items that failed to delete with associated error details.

    :param items: A list of tuples, where each tuple contains the item ID and name to be deleted.
    :param delete_func: A callable responsible for deleting each item, taking the item ID as an argument.
    :param total_items_amount: The total number of items to delete, used for logging progress.
    :param item_type: The type of items being deleted, used in logging messages (defaults to "run").

    :return: A tuple containing:
             - deleted_items: A list of tuples of successfully deleted items, with each tuple containing (name, ID).
             - failed_items: A list of tuples for failed deletions, with each tuple containing
               (name, ID, exception, exception reason).
    """
    deleted_count = 0
    failed_items = []
    deleted_items = []

    for item_id, name in items:
        try:
            delete_func(item_id)
            deleted_count += 1
            deleted_items.append((name, item_id))
            if deleted_count % DELETION_LOG_INTERVAL == 0:
                mlrun.utils.logger.info(
                    f"Deleted {deleted_count}/{total_items_amount} {item_type}s successfully"
                )
                # A 5-second sleep is used to balance KFP load and limit the rate of deletion requests.
                sleep(5)
        except Exception as exc:
            failed_items.append((name, item_id, exc, exc.reason))
            mlrun.utils.logger.warning(
                f"Failed to delete {item_type} '{name}' with ID: {item_id}. Error: {exc}"
            )
    return deleted_items, failed_items


def _log_results(
    context: mlrun.MLClientCtx,
    deleted_items: list[tuple[str, str]],
    failed_items: list[tuple[str, str, Exception, str]],
    item_type: str = "run",
):
    # Log results
    context.log_result(
        key=f"{item_type}s_deleted_count",
        value=len(deleted_items),
    )

    # Log successfully deleted items as a dataset
    if deleted_items:
        df_succeeded = pd.DataFrame(deleted_items, columns=["Name", "ID"])
        context.log_dataset(
            key=f"{item_type}s_deleted_details",
            df=df_succeeded,
        )

    # Log the count of failed deletions
    num_failed = len(failed_items)
    context.log_result(key=f"{item_type}s_failed_count", value=num_failed)

    # Log details of failed deletions if there are any
    if failed_items:
        df_failed = pd.DataFrame(
            failed_items, columns=["Name", "ID", "Exception", "Reason"]
        )
        context.log_dataset(
            key=f"{item_type}s_failed_details",
            df=df_failed,
        )
