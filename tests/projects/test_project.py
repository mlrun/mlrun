# Copyright 2018 Iguazio
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
import pathlib
import shutil
import tempfile
import zipfile

import deepdiff
import inflection
import pytest

import mlrun
import mlrun.errors
import mlrun.projects.project
import tests.conftest


@pytest.fixture()
def context():
    context = pathlib.Path(tests.conftest.tests_root_directory) / "projects" / "test"
    yield context

    # clean up
    if context.exists():
        shutil.rmtree(context)


def test_sync_functions():
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.set_function("hub://describe", "describe")
    project_function_object = project.spec._function_objects
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    project.export(str(project_file_path))
    imported_project = mlrun.load_project("./", str(project_file_path), save=False)
    assert imported_project.spec._function_objects == {}
    imported_project.sync_functions()
    _assert_project_function_objects(imported_project, project_function_object)

    fn = project.get_function("describe")
    assert fn.metadata.name == "describe", "func did not return"

    # test that functions can be fetched from the DB (w/o set_function)
    mlrun.import_function("hub://sklearn_classifier", new_name="train").save()
    fn = project.get_function("train")
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
    project = mlrun.load_project("./", str(legacy_project_file_path), save=False)
    assert project.kind == "project"
    assert project.metadata.name == project_name
    assert project.spec.description == description
    # assert accessible from the project as well
    assert project.description == description
    assert project.spec.artifact_path == artifact_path
    # assert accessible from the project as well
    assert project.artifact_path == artifact_path
    assert (
        deepdiff.DeepDiff(
            params,
            project.spec.params,
            ignore_order=True,
        )
        == {}
    )
    # assert accessible from the project as well
    assert (
        deepdiff.DeepDiff(
            params,
            project.params,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            legacy_project.functions,
            project.functions,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            legacy_project.workflows,
            project.workflows,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            legacy_project.artifacts,
            project.artifacts,
            ignore_order=True,
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
    project = mlrun.projects.project.new_project(project_name, save=False)
    project.export(filepath=project_file_path)


def test_new_project_context_doesnt_exist():
    project_name = "project-name"
    project_dir_path = (
        pathlib.Path(tests.conftest.results) / "new-dir" / "another-new-dir"
    )
    mlrun.projects.project.new_project(project_name, project_dir_path, save=False)


def test_create_project_with_invalid_name():
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.projects.project.new_project(invalid_name, init_git=False, save=False)


def test_get_set_params():
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
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
        project = mlrun.new_project(project_name, user_project=True, save=False)
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


@pytest.mark.parametrize(
    "url,project_name,project_files,clone,num_of_files_to_create,create_child_dir,"
    "override_context,expect_error,error_msg",
    [
        (
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "project.zip",
            "pipe2",
            ["prep_data.py", "project.yaml"],
            True,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "project.tar.gz",
            "pipe2",
            ["prep_data.py", "project.yaml"],
            True,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            "git://github.com/mlrun/project-demo.git",
            "pipe",
            ["prep_data.py", "project.yaml", "kflow.py", "newflow.py"],
            True,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "project.zip",
            "pipe2",
            ["prep_data.py", "project.yaml"],
            False,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "project.tar.gz",
            "pipe2",
            ["prep_data.py", "project.yaml"],
            False,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            "git://github.com/mlrun/project-demo.git",
            "pipe",
            [],
            False,
            3,
            True,
            "",
            True,
            "Failed to load project from git, context directory is not empty. "
            "Set clone param to True to remove the contents of the context directory.",
        ),
        (
            "git://github.com/mlrun/project-demo.git",
            "pipe",
            [],
            False,
            0,
            False,
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "body.txt",
            True,
            "projects/assets/body.txt' already exists and is not an empty directory",
        ),
        (
            "git://github.com/mlrun/project-demo.git",
            "pipe",
            ["prep_data.py", "project.yaml", "kflow.py", "newflow.py"],
            False,
            0,
            False,
            "",
            False,
            "",
        ),
    ],
)
def test_load_project(
    context,
    url,
    project_name,
    project_files,
    clone,
    num_of_files_to_create,
    create_child_dir,
    override_context,
    expect_error,
    error_msg,
):
    temp_files = []
    child_dir = os.path.join(context, "child")

    # use override context to test invalid paths - it will not be deleted on teardown
    context = override_context or context

    # create random files
    if num_of_files_to_create:
        context.mkdir()
        temp_files = [
            tempfile.NamedTemporaryFile(dir=context, delete=False).name
            for _ in range(num_of_files_to_create)
        ]
        for temp_file in temp_files:
            assert os.path.exists(os.path.join(context, temp_file))

    if create_child_dir:
        os.mkdir(child_dir)

    if expect_error:
        with pytest.raises(Exception) as exc:
            mlrun.load_project(context=context, url=url, clone=clone, save=False)
        assert error_msg in str(exc.value)
        return

    project = mlrun.load_project(context=context, url=url, clone=clone, save=False)

    for temp_file in temp_files:

        # verify that the context directory was cleaned if clone is True
        assert os.path.exists(os.path.join(context, temp_file)) is not clone

    if create_child_dir:
        assert os.path.exists(child_dir) is not clone

    assert project.name == project_name
    assert project.spec.context == context
    assert project.spec.source == str(url)
    for project_file in project_files:
        assert os.path.exists(os.path.join(context, project_file))


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


def test_set_func_requirements():
    project = mlrun.projects.MlrunProject("newproj", default_requirements=["pandas"])
    project.set_function("hub://describe", "desc1", requirements=["x"])
    assert project.get_function("desc1", enrich=True).spec.build.commands == [
        "python -m pip install x",
        "python -m pip install pandas",
    ]

    fn = mlrun.import_function("hub://describe")
    project.set_function(fn, "desc2", requirements=["y"])
    assert project.get_function("desc2", enrich=True).spec.build.commands == [
        "python -m pip install y",
        "python -m pip install pandas",
    ]


def test_set_func_with_tag():
    project = mlrun.projects.MlrunProject("newproj", default_requirements=["pandas"])
    project.set_function(
        str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        "desc1",
        tag="v1",
        image="mlrun/mlrun",
    )

    func = project.get_function("desc1")
    assert func.metadata.tag == "v1"
    project.set_function(
        str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        "desc1",
        image="mlrun/mlrun",
    )
    func = project.get_function("desc1")
    assert func.metadata.tag is None
    project.set_function(
        str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        "desc2",
        image="mlrun/mlrun",
    )
    func = project.get_function("desc2")
    assert func.metadata.tag is None


def test_function_run_cli():
    # run function stored in the project spec
    project_dir_path = pathlib.Path(tests.conftest.results) / "project-run-func"
    function_path = pathlib.Path(__file__).parent / "assets" / "handler.py"
    project = mlrun.new_project("run-cli", str(project_dir_path), save=False)
    project.set_function(
        str(function_path),
        "my-func",
        image="mlrun/mlrun",
        handler="myhandler",
    )
    project.export()

    args = "-f my-func --local --dump -p x=3 --out-path '/tmp".split()
    out = tests.conftest.exec_mlrun(args, str(project_dir_path))
    assert out.find("state: completed") != -1, out
    assert out.find("y: 6") != -1, out  # = x * 2


def test_get_artifact_uri():
    project = mlrun.new_project("arti", save=False)
    uri = project.get_artifact_uri("x")
    assert uri == "store://artifacts/arti/x"
    uri = project.get_artifact_uri("y", category="model", tag="prod")
    assert uri == "store://models/arti/y:prod"


def test_export_to_zip():
    project_dir_path = pathlib.Path(tests.conftest.results) / "zip-project"
    project = mlrun.new_project(
        "tozip", context=str(project_dir_path / "code"), save=False
    )
    project.set_function("hub://describe", "desc")
    with (project_dir_path / "code" / "f.py").open("w") as f:
        f.write("print(1)\n")

    zip_path = str(project_dir_path / "proj.zip")
    project.export(zip_path)

    assert os.path.isfile(str(project_dir_path / "code" / "project.yaml"))
    assert os.path.isfile(zip_path)

    zipf = zipfile.ZipFile(zip_path, "r")
    assert set(zipf.namelist()) == set(["./", "f.py", "project.yaml"])

    # check upload to (remote) DataItem
    project.export("memory://x.zip")
    assert mlrun.get_dataitem("memory://x.zip").stat().size


def test_function_receives_project_artifact_path(rundb_mock):
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    mlrun.mlconf.artifact_path = "/tmp"
    proj1 = mlrun.new_project("proj1", save=False)

    # expected to call `get_project`
    mlrun.get_run_db().store_project("proj1", proj1)
    func1 = mlrun.code_to_function(
        "func", kind="job", image="mlrun/mlrun", handler="myhandler", filename=func_path
    )
    run1 = func1.run(local=True)
    assert run1.spec.output_path == mlrun.mlconf.artifact_path
    rundb_mock.reset()

    proj1.spec.artifact_path = "/var"

    func2 = mlrun.code_to_function(
        "func", kind="job", image="mlrun/mlrun", handler="myhandler", filename=func_path
    )
    run2 = func2.run(local=True)
    assert run2.spec.output_path == proj1.spec.artifact_path

    run3 = func2.run(local=True, artifact_path="/not/tmp")
    assert run3.spec.output_path == "/not/tmp"

    # expected to call `get_project`
    mlrun.get_run_db().store_project("proj1", proj1)
    run4 = func2.run(local=True, project="proj1")
    assert run4.spec.output_path == proj1.spec.artifact_path


def test_run_function_passes_project_artifact_path(rundb_mock):
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    mlrun.mlconf.artifact_path = "/tmp"

    proj1 = mlrun.new_project("proj1", save=False)
    proj1.set_function(func_path, "f1", image="mlrun/mlrun", handler="myhandler")

    # expected to call `get_project` because there is no proj1.artifact_path
    mlrun.get_run_db().store_project("proj1", proj1)
    run1 = proj1.run_function("f1", local=True)
    assert run1.spec.output_path == mlrun.mlconf.artifact_path
    rundb_mock.reset()

    proj1.spec.artifact_path = "/var"

    run2 = proj1.run_function("f1", local=True)
    assert run2.spec.output_path == proj1.spec.artifact_path

    mlrun.pipeline_context.workflow_artifact_path = "/data"
    run3 = proj1.run_function("f1", local=True)
    assert run3.spec.output_path == mlrun.pipeline_context.workflow_artifact_path

    # without using project's run_function
    run4 = mlrun.run_function(proj1.get_function("f1"))
    assert run4.spec.output_path == mlrun.pipeline_context.workflow_artifact_path

    # without using project's run_function, but passing project object instead
    run5 = mlrun.run_function(proj1.get_function("f1"), project_object=proj1)
    assert run5.spec.output_path == mlrun.pipeline_context.workflow_artifact_path

    mlrun.pipeline_context.clear()
    # expected to call `get_project`
    mlrun.get_run_db().store_project("proj1", proj1)
    run6 = mlrun.run_function(proj1.get_function("f1"), project_object=proj1)
    assert run6.spec.output_path == proj1.spec.artifact_path


def test_project_ops():
    # verify that project ops (run_function, ..) will use the right project (and not the pipeline_context)
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    proj1 = mlrun.new_project("proj1", save=False)
    proj1.spec.artifact_path = "/tmp"
    proj1.set_function(func_path, "f1", image="mlrun/mlrun", handler="myhandler")

    proj2 = mlrun.new_project("proj2", save=False)
    proj2.spec.artifact_path = "/tmp"
    proj2.set_function(func_path, "f2", image="mlrun/mlrun", handler="myhandler")

    run = proj1.run_function("f1", params={"x": 1}, local=True)
    assert run.spec.function.startswith("proj1/f1")
    assert run.output("y") == 2  # = x * 2

    run = proj2.run_function("f2", params={"x": 2}, local=True)
    assert run.spec.function.startswith("proj2/f2")
    assert run.output("y") == 4  # = x * 2
