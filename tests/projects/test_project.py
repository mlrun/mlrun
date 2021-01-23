import getpass
import pathlib
import pytest
import os

import deepdiff

import mlrun
import mlrun.errors
import mlrun.projects.project
import tests.conftest


def test_sync_functions():
    project_name = "project-name"
    project = mlrun.new_project(project_name)
    project.set_function("hub://describe")
    project_function_object = project.spec._function_objects
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    project.export(str(project_file_path))
    imported_project = mlrun.load_project(None, str(project_file_path))
    assert imported_project.spec._function_objects == {}
    imported_project.sync_functions()
    _assert_project_function_objects(imported_project, project_function_object)


def _assert_project_function_objects(project, expected_function_objects):
    project_function_objects = project.spec._function_objects
    assert len(project_function_objects) == len(expected_function_objects)
    for function_name, function_object in expected_function_objects.items():
        assert function_name in project_function_objects
        assert (
            deepdiff.DeepDiff(
                project_function_objects[function_name].to_dict(),
                function_object.to_dict(),
                ignore_order=True,
                exclude_paths=["root['spec']['build']['code_origin']"],
            )
            == {}
        )


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
    # assert accessible from the project as well
    assert project.description == description
    assert project.spec.artifact_path == artifact_path
    assert deepdiff.DeepDiff(params, project.spec.params, ignore_order=True,) == {}
    # assert accessible from the project as well
    assert deepdiff.DeepDiff(params, project.params, ignore_order=True,) == {}
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


def test_create_project_with_invalid_name():
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.projects.project.new_project(invalid_name, init_git=False)


def test_get_set_params():
    project_name = "project-name"
    project = mlrun.new_project(project_name)
    param_key = "param-key"
    param_value = "param-value"
    project.params[param_key] = param_value
    assert param_value == project.get_param(param_key)
    default_value = "default-value"
    assert project.get_param("not-exist", default_value) == default_value


def test_user_project():
    project_name = "project-name"
    user = os.environ.get("V3IO_USERNAME") or getpass.getuser()
    project = mlrun.new_project(project_name, user_project=True)
    assert (
        project.metadata.name == f"{project_name}-{user}"
    ), "project name doesnt include user name"
