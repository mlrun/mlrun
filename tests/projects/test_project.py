import os
import pathlib

import deepdiff
import inflection
import pytest

import mlrun
import mlrun.errors
import mlrun.projects.project
import tests.conftest


def test_sync_functions():
    project_name = "project-name"
    project = mlrun.new_project(project_name)
    project.set_function("hub://describe", "describe")
    project_function_object = project.spec._function_objects
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    project.export(str(project_file_path))
    imported_project = mlrun.load_project("./", str(project_file_path))
    assert imported_project.spec._function_objects == {}
    imported_project.sync_functions()
    _assert_project_function_objects(imported_project, project_function_object)

    fn = project.func("describe")
    assert fn.metadata.name == "describe", "func did not return"

    # test that functions can be fetched from the DB (w/o set_function)
    mlrun.import_function("hub://sklearn_classifier", new_name="train").save()
    fn = project.func("train")
    assert fn.metadata.name == "train", "train func did not return"


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
    project = mlrun.load_project("./", str(legacy_project_file_path))
    assert project.kind == "project"
    assert project.metadata.name == project_name
    assert project.spec.description == description
    # assert accessible from the project as well
    assert project.description == description
    assert project.spec.artifact_path == artifact_path
    # assert accessible from the project as well
    assert project.artifact_path == artifact_path
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


def test_export_project_dir_doesnt_exist():
    project_name = "project-name"
    project_file_path = (
        pathlib.Path(tests.conftest.results)
        / "new-dir"
        / "another-new-dir"
        / "project.yaml"
    )
    project = mlrun.projects.project.new_project(project_name)
    project.export(filepath=project_file_path)


def test_new_project_context_doesnt_exist():
    project_name = "project-name"
    project_dir_path = (
        pathlib.Path(tests.conftest.results) / "new-dir" / "another-new-dir"
    )
    mlrun.projects.project.new_project(project_name, project_dir_path)


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
    original_username = os.environ.get("V3IO_USERNAME")
    usernames = ["valid-username", "require_Normalization"]
    for username in usernames:
        os.environ["V3IO_USERNAME"] = username
        project = mlrun.new_project(project_name, user_project=True)
        assert (
            project.metadata.name
            == f"{project_name}-{inflection.dasherize(username.lower())}"
        ), "project name doesnt include user name"
    if original_username is not None:
        os.environ["V3IO_USERNAME"] = original_username


def test_build_project_from_minimal_dict():
    # When mlrun is follower, the created project will usually have all values set to None when created from the leader
    # API, verify we successfully initialize Project instance from that
    project_dict = {
        "metadata": {"name": "default", "labels": None, "annotations": None},
        "spec": {
            "description": None,
            "goals": None,
            "params": None,
            "functions": None,
            "workflows": None,
            "artifacts": None,
            "artifact_path": None,
            "conda": None,
            "source": None,
            "subpath": None,
            "origin_url": None,
            "desired_state": "online",
        },
        "status": {"state": "online"},
    }
    mlrun.projects.MlrunProject.from_dict(project_dict)


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
