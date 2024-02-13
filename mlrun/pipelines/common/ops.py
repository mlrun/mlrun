import inflection

import mlrun

from mlrun.pipelines.common.helpers import project_annotation, run_annotation, function_annotation
from mlrun.utils import get_in, logger


def format_summary_from_kfp_run(
    kfp_run, project=None, run_db: "mlrun.db.RunDBInterface" = None
):
    override_project = project if project and project != "*" else None
    dag, project, message = generate_kfp_dag_and_resolve_project(
        kfp_run, override_project
    )
    run_id = kfp_run.id
    logger.debug("Formatting summary from KFP run", run_id=run_id, project=project)

    # run db parameter allows us to use the same db session for the whole flow and avoid session isolation issues
    if not run_db:
        run_db = mlrun.db.get_run_db()

    # enrich DAG with mlrun run info
    runs = run_db.list_runs(project=project, labels=f"workflow={run_id}")

    for run in runs:
        step = get_in(run, ["metadata", "labels", "mlrun/runner-pod"])
        if step and step in dag:
            dag[step]["run_uid"] = get_in(run, "metadata.uid")
            dag[step]["kind"] = get_in(run, "metadata.labels.kind")
            error = get_in(run, "status.error")
            if error:
                dag[step]["error"] = error

    short_run = {
        "graph": dag,
        "run": mlrun.utils.helpers.format_run(kfp_run),
    }
    short_run["run"]["project"] = project
    short_run["run"]["message"] = message
    logger.debug("Completed summary formatting", run_id=run_id, project=project)
    return short_run


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest
    if not workflow:
        return None, project, None

    templates = {}
    for template in workflow["spec"]["templates"]:
        project = project or get_in(
            template, ["metadata", "annotations", project_annotation], ""
        )
        name = template["name"]
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", run_annotation], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", function_annotation], ""
            ),
        }

    nodes = workflow["status"].get("nodes", {})
    dag = {}
    for node in nodes.values():
        name = node["displayName"]
        record = {
            k: node[k] for k in ["phase", "startedAt", "finishedAt", "type", "id"]
        }

        # snake case
        # align kfp fields to mlrun snake case convention
        # create snake_case for consistency.
        # retain the camelCase for compatibility
        for key in list(record.keys()):
            record[inflection.underscore(key)] = record[key]

        record["parent"] = node.get("boundaryID", "")
        record["name"] = name
        record["children"] = node.get("children", [])
        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[node["id"]] = record

    return dag, project, workflow["status"].get("message", "")
