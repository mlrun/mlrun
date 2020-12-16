import pathlib

import deepdiff

import mlrun
import mlrun.projects.project
import tests.conftest


def test_create_project_from_file_with_legacy_structure():
    project_name = "project-name"
    description = "project description"
    params = {"param_key": "param value"}
    artifact_path = "/tmp"
    legacy_project = mlrun.projects.project.MlrunProjectLegacy(
        project_name, description, params, artifact_path=artifact_path
    )
    function_name = "trainer-function"
    function = mlrun.new_function(function_name, project_name)
    legacy_project.set_function(function, function_name)
    legacy_project.set_function("hub://describe", "describe")
    workflow_name = "workflow-name"
    workflow_file_path = (
        pathlib.Path(tests.conftest.tests_root_directory) / "projects" / "workflow.py"
    )
    legacy_project.set_workflow(workflow_name, str(workflow_file_path))
    artifact_dict = {
        "key": "raw-data",
        "kind": "",
        "iter": 0,
        "tree": "latest",
        "target_path": "https://raw.githubusercontent.com/mlrun/demos/master/customer-churn-prediction/WA_Fn-UseC_-Telc"
        "o-Customer-Churn.csv",
        "db_key": "raw-data",
    }
    legacy_project.artifacts = [artifact_dict]
    legacy_project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    legacy_project.save(str(legacy_project_file_path))
    project = mlrun.load_project(None, str(legacy_project_file_path))
    assert project.kind == "project"
    assert project.metadata.name == project_name
    assert project.spec.description == description
    assert project.spec.artifact_path == artifact_path
    assert deepdiff.DeepDiff(params, project.spec.params, ignore_order=True,) == {}
    assert (
        deepdiff.DeepDiff(
            legacy_project.functions, project.functions, ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            legacy_project.workflows, project.workflows, ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            legacy_project.artifacts, project.artifacts, ignore_order=True,
        )
        == {}
    )
