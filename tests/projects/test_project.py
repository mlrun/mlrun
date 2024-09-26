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

import os
import os.path
import pathlib
import re
import shutil
import tempfile
import unittest.mock
import zipfile
from contextlib import nullcontext as does_not_raise

import deepdiff
import inflection
import pytest
from mlrun_pipelines.common.models import RunStatuses

import mlrun
import mlrun.alerts.alert
import mlrun.artifacts
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring as mm_consts
import mlrun.db.nopdb
import mlrun.errors
import mlrun.projects.project
import mlrun.runtimes.base
import mlrun.runtimes.nuclio.api_gateway
import mlrun.utils.helpers
import tests.conftest


@pytest.fixture()
def context():
    context = pathlib.Path(tests.conftest.tests_root_directory) / "projects" / "test"
    yield context

    # clean up
    if context.exists():
        shutil.rmtree(context)


def assets_path():
    return pathlib.Path(__file__).absolute().parent / "assets"


def test_sync_functions(rundb_mock):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.set_function("hub://describe", "describe")
    project_function_object = project.spec._function_objects
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    project.export(str(project_file_path))
    imported_project = mlrun.load_project(
        "./", str(project_file_path), save=False, allow_cross_project=True
    )
    assert imported_project.spec._function_objects == {}
    imported_project.sync_functions()
    _assert_project_function_objects(imported_project, project_function_object)

    fn = project.get_function("describe")
    assert fn.metadata.name == "describe", "func did not return"

    # test that functions can be fetched from the DB (w/o set_function)
    mlrun.import_function("hub://auto-trainer", new_name="train").save()
    fn = project.get_function("train")
    assert fn.metadata.name == "train", "train func did not return"


def test_sync_functions_with_names_different_than_default(rundb_mock):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)

    describe_func = mlrun.import_function("hub://describe")
    # set a different name than the default
    project.set_function(describe_func, "new_describe_func")

    project_function_object = project.spec._function_objects
    project_function_definition = project.spec._function_definitions

    # sync functions - expected to sync the function objects from the definitions
    project.sync_functions()

    assert project.spec._function_objects == project_function_object
    assert project.spec._function_definitions == project_function_definition


def test_sync_functions_preserves_existing(rundb_mock):
    project = mlrun.new_project("project-name", save=False)
    project.set_function("hub://describe", "describe")
    project.set_function("hub://auto-trainer", "auto-trainer")

    old_trainer = project.spec._function_objects.pop("auto-trainer")
    old_describe = project.spec._function_objects["describe"]

    project.sync_functions(always=False)
    assert old_trainer is not project.spec._function_objects["auto-trainer"]
    assert old_trainer is not project.get_function("auto-trainer")
    assert old_describe is project.spec._function_objects["describe"]
    assert old_describe is project.get_function("describe")

    project._initialized = False
    project.sync_functions(always=True)
    assert old_describe is not project.spec._function_objects["describe"]
    assert old_describe is not project.get_function("describe")


def test_sync_functions_unavailable_file():
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.spec._function_definitions["non-existing-function"] = {
        "handler": "func",
        "image": "mlrun/mlrun",
        "kind": "job",
        "name": "func",
        "url": "func.py",
    }
    with pytest.raises(mlrun.errors.MLRunMissingDependencyError):
        project.sync_functions()


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
    "url,project_name,project_files,clone,allow_cross_project,num_of_files_to_create,create_child_dir,"
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
            / "project.zip",
            "different1name",
            ["prep_data.py", "project.yaml"],
            True,
            False,
            3,
            True,
            "",
            True,
            "Project name mismatch",
        ),
        (
            pathlib.Path(tests.conftest.tests_root_directory)
            / "projects"
            / "assets"
            / "project.zip",
            "different2name",
            ["prep_data.py", "project.yaml"],
            True,
            None,
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
            "different1name",
            ["prep_data.py", "project.yaml"],
            True,
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
            False,
            3,
            True,
            "",
            False,
            "",
        ),
        (
            "git://github.com/mlrun/project-demo.git#refs/heads/main",
            "pipe",
            ["prep_data.py", "project.yaml", "kflow.py", "newflow.py"],
            True,
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
            / "project.zip",
            "pipe2",
            ["prep_data.py", "project.yaml"],
            False,
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
            "git://github.com/mlrun/project-demo.git#refs/heads/main",
            "pipe",
            ["prep_data.py", "project.yaml", "kflow.py", "newflow.py"],
            False,
            False,
            0,
            False,
            "",
            False,
            "",
        ),
        (
            "ssh://git@something/something",
            "something",
            [],
            False,
            False,
            0,
            False,
            "",
            True,
            "Unsupported url scheme, supported schemes are: git://, db:// or "
            ".zip/.tar.gz/.yaml file path (could be local or remote) or project name which will be loaded from DB",
        ),
    ],
)
def test_load_project(
    context,
    url,
    project_name,
    project_files,
    clone,
    allow_cross_project,
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
            mlrun.load_project(
                context=context,
                url=url,
                clone=clone,
                save=False,
                name=project_name,
                allow_cross_project=allow_cross_project,
            )
        assert error_msg in str(exc.value)
        return

    project = mlrun.load_project(
        context=context,
        url=url,
        clone=clone,
        save=False,
        name=project_name,
        allow_cross_project=allow_cross_project,
    )

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


@pytest.mark.parametrize(
    "from_template,project_name,override_context,expect_error,error_msg",
    [
        (
            str(
                pathlib.Path(tests.conftest.tests_root_directory)
                / "projects"
                / "assets"
                / "project.zip"
            ),
            "different1name",
            "",
            False,
            "",
        ),
    ],
)
def test_new_project(
    context,
    from_template,
    project_name,
    override_context,
    expect_error,
    error_msg,
):
    # use override context to test invalid paths - it will not be deleted on teardown
    context = override_context or context

    if expect_error:
        with pytest.raises(Exception) as exc:
            mlrun.new_project(
                context=context,
                from_template=from_template,
                save=False,
                name=project_name,
            )
        assert error_msg in str(exc.value)
        return

    project = mlrun.new_project(
        context=context,
        from_template=from_template,
        save=False,
        name=project_name,
    )

    assert project.name == project_name


@pytest.mark.parametrize("op", ["new", "load"])
def test_project_with_setup(context, op):
    # load the project from the "assets/load_setup_test" dir, and init using the project_setup.py in it
    project_path = (
        pathlib.Path(tests.conftest.tests_root_directory)
        / "projects"
        / "assets"
        / f"{op}_setup_test"
    )
    name = f"projset-{op}"
    if op == "new":
        func = mlrun.new_project
    else:
        func = mlrun.load_project
    project = func(
        context=project_path, name=name, save=False, parameters={"p2": "123"}
    )
    mlrun.utils.logger.info("Created project", project=project)

    # see assets/load_setup_test/project_setup.py for extra project settings
    # test that a function was added and its metadata was set from param[p2]
    prep_func = project.get_function("prep-data")
    assert prep_func.metadata.labels == {"tst1": "123"}  # = p2

    # test that a serving function was set with a graph element (model)
    srv_func = project.get_function("serving")
    assert srv_func.spec.graph["x"].class_name == "MyCls", "serving graph was not set"

    # test that the project metadata was set correctly
    assert project.name == name
    assert project.spec.context == project_path

    # test that the params contain all params from the yaml, the load, and the setup script
    if op == "new":
        assert project.spec.params == {"p2": "123", "test123": "456"}  # no YAML
    else:
        assert project.spec.params == {"p1": "xyz", "p2": "123", "test123": "456"}


@pytest.mark.parametrize(
    "setup_file_contents, exception",
    [
        (b"def setup(project): return 5", pytest.raises(Exception)),
        (b"def setup(project): pass", pytest.raises(Exception)),
        (b"def setup(project): return None", pytest.raises(Exception)),
        (b"def setup(project): return project", does_not_raise()),
    ],
)
def test_project_setup_must_return_project_object(
    context, setup_file_contents, exception
):
    mlrun_project = mlrun.new_project(context=context, name="projset", save=False)
    with tempfile.NamedTemporaryFile(dir=context, delete=False, suffix=".py") as fp:
        fp.write(setup_file_contents)

        # ensure the file is written, so the setup will be imported properly
        fp.flush()
        with exception as exc:
            mlrun.projects.project._run_project_setup(
                mlrun_project, fp.name, save=False
            )
        if exc:
            assert "must return a project object" in str(exc.value)


@pytest.mark.parametrize(
    "sync,expected_num_of_funcs, save",
    [
        (
            False,
            0,
            False,
        ),
        (
            True,
            5,
            False,
        ),
        (
            True,
            5,
            True,
        ),
    ],
)
def test_load_project_and_sync_functions(
    context, rundb_mock, sync, expected_num_of_funcs, save
):
    url = "git://github.com/mlrun/project-demo.git"
    project = mlrun.load_project(
        context=str(context),
        url=url,
        sync_functions=sync,
        save=save,
        allow_cross_project=True,
    )
    assert len(project.spec._function_objects) == expected_num_of_funcs

    if sync:
        function_names = project.spec._function_definitions.keys()
        assert len(function_names) == expected_num_of_funcs
        for func in function_names:
            fn = project.get_function(func)
            normalized_name = mlrun.utils.helpers.normalize_name(func)
            assert fn.metadata.name == normalized_name, "func did not return"

            if save:
                assert normalized_name in rundb_mock._functions


def _assert_project_function_objects(project, expected_function_objects):
    project_function_objects = project.spec._function_objects
    assert len(project_function_objects) == len(expected_function_objects)
    for function_name, function_object in expected_function_objects.items():
        assert function_name in project_function_objects
        project_function = project_function_objects[function_name].to_dict()
        project_function["metadata"]["tag"] = (
            project_function["metadata"]["tag"] or "latest"
        )
        assert (
            deepdiff.DeepDiff(
                project_function,
                function_object.to_dict(),
                ignore_order=True,
                exclude_paths=["root['spec']['build']['code_origin']"],
            )
            == {}
        )


def test_set_function_requirements(rundb_mock):
    project = mlrun.projects.project.MlrunProject.from_dict(
        {
            "metadata": {
                "name": "newproj",
            },
            "spec": {
                "default_requirements": ["pandas>1, <3"],
            },
        }
    )
    project.set_function("hub://describe", "desc1", requirements=["x"])
    assert project.get_function("desc1", enrich=True).spec.build.requirements == [
        "x",
        "pandas>1, <3",
    ]

    fn = mlrun.import_function("hub://describe")
    project.set_function(fn, "desc2", requirements=["y"])
    assert project.get_function("desc2", enrich=True).spec.build.requirements == [
        "y",
        "pandas>1, <3",
    ]


def test_backwards_compatibility_get_non_normalized_function_name(rundb_mock):
    project = mlrun.projects.MlrunProject(
        mlrun.ProjectMetadata("project"),
        mlrun.projects.ProjectSpec(default_requirements=["pandas>1, <3"]),
    )
    func_name = "name_with_underscores"
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")

    func = mlrun.code_to_function(
        name=func_name,
        kind="job",
        image="mlrun/mlrun",
        handler="myhandler",
        filename=func_path,
    )
    # nuclio also normalizes the name, so we de-normalize the function name before storing it
    func.metadata.name = func_name

    # mock the normalize function response in order to insert a non-normalized function name to the db
    with unittest.mock.patch("mlrun.utils.normalize_name", return_value=func_name):
        project.set_function(name=func_name, func=func)

    # getting the function using the original non-normalized name, and ensure that querying it works
    enriched_function = project.get_function(key=func_name)
    assert enriched_function.metadata.name == func_name

    enriched_function = project.get_function(key=func_name, sync=True)
    assert enriched_function.metadata.name == func_name

    # override the function by sending an update request,
    # a new function is created, and the old one is no longer accessible
    normalized_function_name = mlrun.utils.normalize_name(func_name)
    func.metadata.name = normalized_function_name
    project.set_function(name=func_name, func=func)

    # using both normalized and non-normalized names to query the function
    enriched_function = project.get_function(key=normalized_function_name)
    assert enriched_function.metadata.name == normalized_function_name

    resp = project.get_function(key=func_name)
    assert resp.metadata.name == normalized_function_name


def test_set_function_underscore_name(rundb_mock):
    project = mlrun.projects.MlrunProject(
        mlrun.projects.ProjectMetadata("project"),
        mlrun.projects.ProjectSpec(default_requirements=["pandas>1, <3"]),
    )
    func_name = "name_with_underscores"

    # create a function with a name that includes underscores
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    func = mlrun.code_to_function(
        name=func_name,
        kind="job",
        image="mlrun/mlrun",
        handler="myhandler",
        filename=func_path,
    )
    project.set_function(name=func_name, func=func)

    # get the function using the original name (with underscores) and ensure that it works and returns normalized name
    normalized_name = mlrun.utils.normalize_name(func_name)
    enriched_function = project.get_function(key=func_name)
    assert enriched_function.metadata.name == normalized_name

    # get the function using a normalized name and make sure it works
    enriched_function = project.get_function(key=normalized_name)
    assert enriched_function.metadata.name == normalized_name


def test_set_func_with_tag():
    project = mlrun.projects.project.MlrunProject.from_dict(
        {
            "metadata": {
                "name": "newproj",
            },
            "spec": {
                "default_requirements": ["pandas"],
            },
        }
    )
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
    assert func.metadata.tag == "latest"
    project.set_function(
        str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        "desc2",
        image="mlrun/mlrun",
    )
    func = project.get_function("desc2")
    assert func.metadata.tag == "latest"


def test_set_function_with_tagged_key():
    project = mlrun.new_project("set-func-tagged-key", save=False)
    # create 2 functions with different tags
    tag_v1 = "v1"
    tag_v2 = "v2"
    my_func_v1 = mlrun.code_to_function(
        filename=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        kind="job",
        tag=tag_v1,
    )
    my_func_v2 = mlrun.code_to_function(
        filename=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        kind="job",
        name="my_func",
        tag=tag_v2,
    )

    # set the functions
    # function key is <function name> ("handler")
    project.set_function(my_func_v1)
    # function key is <function name>:<tag> ("handler:v1")
    project.set_function(my_func_v1, tag=tag_v1)
    # function key is "my_func"
    project.set_function(my_func_v2, name=my_func_v2.metadata.name)
    # function key is "my_func:v2"
    project.set_function(my_func_v2, name=f"{my_func_v2.metadata.name}:{tag_v2}")

    assert len(project.spec._function_objects) == 4

    func = project.get_function(f"{my_func_v1.metadata.name}:{tag_v1}")
    assert func.metadata.tag == tag_v1

    func = project.get_function(my_func_v1.metadata.name, tag=tag_v1)
    assert func.metadata.tag == tag_v1

    func = project.get_function(my_func_v1.metadata.name)
    assert func.metadata.tag == tag_v1

    func = project.get_function(my_func_v2.metadata.name)
    assert func.metadata.tag == tag_v2

    func = project.get_function(f"{my_func_v2.metadata.name}:{tag_v2}")
    assert func.metadata.tag == tag_v2

    func = project.get_function(my_func_v2.metadata.name, tag=tag_v2)
    assert func.metadata.tag == tag_v2

    func = project.get_function(f"{my_func_v2.metadata.name}:{tag_v2}", tag=tag_v2)
    assert func.metadata.tag == tag_v2


def test_set_function_update_code():
    project = mlrun.new_project("set-func-update-code", save=False)
    for i in range(2):
        func = project.set_function(
            func=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
            name="handler",
            kind="job",
            image="mlrun/mlrun",
            handler="myhandler",
            tag="v1",
        )

        assert id(func) == id(
            project.get_function("handler")
        ), f"Function of index {i} was not set correctly"
        assert id(func) == id(
            project.get_function("handler:v1")
        ), f"Function of index {i} was not set and tagged correctly"


def test_set_function_with_conflicting_tag():
    project = mlrun.new_project("set-func-conflicting-tag", save=False)
    with pytest.raises(ValueError) as exc:
        project.set_function(
            func=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
            name="handler:v2",
            kind="job",
            image="mlrun/mlrun",
            handler="myhandler",
            tag="v1",
        )
    assert "Tag parameter (v1) and tag in function name (handler:v2) must match" in str(
        exc.value
    )


def test_set_function_with_multiple_tags():
    project = mlrun.new_project("set-func-multi-tags", save=False)
    name = "handler:v2:v3"
    with pytest.raises(ValueError) as exc:
        project.set_function(
            func=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
            name=name,
            kind="job",
            image="mlrun/mlrun",
            handler="myhandler",
        )
    assert (
        f"Function name ({name}) must be in the format <name>:<tag> or <name>"
        in str(exc.value)
    )


@pytest.mark.parametrize(
    "name, tag, expected_name, expected_tag",
    [
        ("handler:v2", None, "handler", "v2"),
        ("handler", None, "handler", "latest"),
        ("handler", "v2", "handler", "v2"),
    ],
)
def test_set_function_name_and_tag(name, tag, expected_name, expected_tag):
    project = mlrun.new_project("set-func-untagged-name", save=False)
    func = project.set_function(
        func=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        name=name,
        kind="job",
        image="mlrun/mlrun",
        handler="myhandler",
        tag=tag,
    )
    assert func.metadata.name == expected_name
    assert func.metadata.tag == expected_tag


def test_set_function_from_object():
    project = mlrun.new_project("set-func-from-object", save=False)
    func = mlrun.code_to_function(
        filename=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        kind="job",
        name="handler",
        image="mlrun/mlrun",
        handler="myhandler",
        tag="v1",
    )
    project_function = project.set_function(func)
    assert project_function.metadata.name == "handler"
    assert project_function.metadata.tag == "v1"

    project_function = project.get_function("handler")
    assert project_function.metadata.name == "handler"
    assert project_function.metadata.tag == "v1"

    assert "handler:v1" not in project.spec._function_objects


def test_set_function_from_object_override_tag():
    project = mlrun.new_project("set-func-from-object", save=False)
    func = mlrun.code_to_function(
        filename=str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
        kind="job",
        name="handler",
        image="mlrun/mlrun",
        handler="myhandler",
        tag="v1",
    )
    project_function_v2 = project.set_function(func, tag="v2")
    assert project_function_v2.metadata.name == "handler"
    assert project_function_v2.metadata.tag == "v2"

    project_function_v2 = project.get_function("handler")
    assert project_function_v2.metadata.name == "handler"
    assert project_function_v2.metadata.tag == "v2"

    project_function_v2 = project.get_function("handler:v2")
    assert project_function_v2.metadata.name == "handler"
    assert project_function_v2.metadata.tag == "v2"

    project_function_v3 = project.set_function(func, name="other-func:v3")
    assert project_function_v3.metadata.name == "other-func"
    assert project_function_v3.metadata.tag == "v3"

    project_function_v3 = project.get_function("other-func:v3")
    assert project_function_v3.metadata.name == "other-func"
    assert project_function_v3.metadata.tag == "v3"

    # only name param (other-func:v3) should be set
    assert "other-func" not in project.spec._function_objects

    # assert original function changed
    assert func.metadata.name == "other-func"
    assert func.metadata.tag == "v3"


def test_set_function_with_relative_path(context):
    project = mlrun.new_project("inline", context=str(assets_path()), save=False)

    project.set_function(
        "handler.py",
        "desc1",
        image="mlrun/mlrun",
    )

    func = project.get_function("desc1")
    assert func is not None and func.spec.build.origin_filename.startswith(
        str(assets_path())
    )


@pytest.mark.parametrize(
    "artifact_path,file_exists,expectation",
    [
        ("handler.py", True, does_not_raise()),
        ("handler.py", False, pytest.raises(OSError)),
    ],
)
def test_set_artifact_validates_file_exists(
    monkeypatch, artifact_path, file_exists, expectation
):
    artifact_key = "my-artifact"
    project = mlrun.new_project("inline", context=str(assets_path()), save=False)

    monkeypatch.setattr(
        os.path,
        "isfile",
        lambda path: path == str(assets_path() / artifact_path) and file_exists,
    )

    with expectation:
        project.set_artifact(
            artifact_key,
            artifact_path,
        )
        assert project.spec.artifacts[0]["key"] == artifact_key
        assert project.spec.artifacts[0]["import_from"] == str(
            assets_path() / artifact_path
        )


def test_import_artifact_using_relative_path():
    project = mlrun.new_project("inline", context=str(assets_path()), save=False)

    # log an artifact and save the content/body in the object (inline)
    artifact = project.log_artifact(
        "x", body="123", is_inline=True, artifact_path=str(assets_path())
    )
    assert artifact.spec.get_body() == "123"
    artifact.export(f"{str(assets_path())}/artifact.yaml")

    # importing the artifact using a relative path
    artifact = project.import_artifact("artifact.yaml", "y")
    assert artifact.spec.get_body() == "123"
    assert artifact.metadata.key == "y"
    assert artifact.spec.db_key == "y"


def test_import_artifact_retain_producer(rundb_mock):
    base_path = tests.conftest.results
    project_1 = mlrun.new_project(
        name="project-1", context=f"{base_path}/project_1", save=False
    )
    project_2 = mlrun.new_project(
        name="project-2", context=f"{base_path}/project_2", save=False
    )

    # set project owners
    project_1.spec.owner = "owner-1"

    # create an artifact with a 'run' producer
    artifact = mlrun.artifacts.Artifact(key="x", body="123", is_inline=True)
    run_name = "my-run"
    run_tag = "sometag123"

    # we set the producer as dict so the export will work
    artifact.producer = mlrun.artifacts.ArtifactProducer(
        kind="run",
        project=project_1.name,
        name=run_name,
        owner=project_1.spec.owner,
    ).get_meta()

    # imitate the artifact being produced by a run with uri and without a tag
    artifact.producer["uri"] = f"{project_1.name}/{run_tag}"
    artifact.producer["project"] = project_1.name

    # the uri is parsed when importing the artifact, so we set the expected producer
    # also, the project is removed from the producer when importing
    expected_producer = {
        "kind": "run",
        "name": run_name,
        "tag": run_tag,
        "owner": project_1.spec.owner,
    }

    # export the artifact
    artifact_path = f"{base_path}/my-artifact.yaml"
    artifact.export(artifact_path)

    # import the artifact to another project
    new_key = "y"
    imported_artifact = project_2.import_artifact(artifact_path, new_key)
    assert imported_artifact.producer == expected_producer

    # set the artifact on the first project
    project_1.set_artifact(artifact.key, artifact)
    project_1.save()

    # load a new project from the first project's context
    project_3 = mlrun.load_project(
        name="project-3", context=project_1.context, allow_cross_project=True
    )

    # make sure the artifact was registered with the new key
    loaded_artifact = project_3.get_artifact(new_key)
    assert loaded_artifact.producer == expected_producer


def test_replace_exported_artifact_producer(rundb_mock):
    base_path = tests.conftest.results
    project_1 = mlrun.new_project(
        name="project-1", context=f"{base_path}/project_1", save=False
    )
    project_2 = mlrun.new_project(
        name="project-2", context=f"{base_path}/project_2", save=False
    )

    # create an artifact with a 'project' producer
    key = "x"
    artifact = mlrun.artifacts.Artifact(key=key, body="123", is_inline=True)

    # we set the producer as dict so the export will work
    artifact.producer = mlrun.artifacts.ArtifactProducer(
        kind="project",
        project=project_1.name,
        name=project_1.name,
    ).get_meta()

    # export the artifact
    artifact_path = f"{base_path}/my-artifact.yaml"
    artifact.export(artifact_path)

    # import the artifact to another project
    new_key = "y"
    imported_artifact = project_2.import_artifact(artifact_path, new_key)
    assert imported_artifact.producer != artifact.producer
    assert imported_artifact.producer["name"] == project_2.name

    # set the artifact on the first project
    project_1.set_artifact(artifact.key, artifact)
    project_1.save()

    # load a new project from the first project's context
    project_3 = mlrun.load_project(
        name="project-3", context=project_1.context, allow_cross_project=True
    )

    # make sure the artifact was registered with the new project producer
    loaded_artifact = project_3.get_artifact(key)
    assert loaded_artifact.producer != artifact.producer
    assert loaded_artifact.producer["name"] == project_3.name


@pytest.mark.parametrize(
    "project_owner,username",
    [
        ("project-owner", None),
        (None, "username"),
        ("project-owner", "username"),
        (None, None),
    ],
)
def test_artifact_owner(
    rundb_mock, project_owner, username, monkeypatch: pytest.MonkeyPatch
):
    if username:
        monkeypatch.setenv("V3IO_USERNAME", username)

    project = mlrun.new_project("artifact-owner", save=False)
    project.spec.owner = project_owner
    artifact = project.log_artifact("x", body="123", format="txt")
    if username:
        assert artifact.producer.get("owner") == username
    else:
        assert artifact.producer.get("owner") == project_owner


@pytest.mark.parametrize(
    "relative_artifact_path,project_context,expected_path,expected_in_context",
    [
        (
            "artifact.yml",
            "/project/context/assets",
            "/project/context/assets/artifact.yml",
            True,
        ),
        (
            "../../artifact.yml",
            "/project/assets/project/context",
            "/project/assets/artifact.yml",
            True,
        ),
        ("../artifact.json", "/project/context", "/project/artifact.json", True),
        ("v3io://artifact.zip", "/project/context", "v3io://artifact.zip", False),
        ("/artifact.json", "/project/context", "/artifact.json", False),
    ],
)
def test_get_item_absolute_path(
    relative_artifact_path, project_context, expected_path, expected_in_context
):
    with unittest.mock.patch("os.path.isfile", return_value=True):
        project = mlrun.new_project("inline", save=False)
        project.spec.context = project_context
        result, in_context = project.get_item_absolute_path(relative_artifact_path)
        assert result == expected_path and in_context == expected_in_context


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

    args = "-f my-func --local --dump -p x=3".split()
    out = tests.conftest.exec_mlrun(args, str(project_dir_path))
    assert out.find("state: completed") != -1, out
    assert out.find("y: 6") != -1, out  # = x * 2


def test_get_artifact_uri():
    project = mlrun.new_project("arti", save=False)
    uri = project.get_artifact_uri("x")
    assert uri == "store://artifacts/arti/x"
    uri = project.get_artifact_uri("y", category="model", tag="prod")
    assert uri == "store://models/arti/y:prod"


def test_export_to_zip(rundb_mock):
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
    assert set(zipf.namelist()) == {"./", "f.py", "project.yaml"}

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
    # because there is not artifact path in the project, then the default artifact path is used
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

    rundb_mock.reset()
    mlrun.pipeline_context.clear(with_project=True)

    func3 = mlrun.code_to_function(
        "func", kind="job", image="mlrun/mlrun", handler="myhandler", filename=func_path
    )
    # expected to call `get_project`, but the project wasn't saved yet, so it will use the default artifact path
    run5 = func3.run(local=True, project="proj1")
    assert run5.spec.output_path == mlrun.mlconf.artifact_path

    proj1.set_function(func_path, "func", kind="job", image="mlrun/mlrun")
    run = proj1.run_function("func", local=True)
    assert run.spec.output_path == proj1.spec.artifact_path

    run = proj1.run_function("func", local=True, artifact_path="/not/tmp")
    assert run.spec.output_path == "/not/tmp"


def test_function_receives_project_default_image():
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    mlrun.mlconf.artifact_path = "/tmp"
    proj1 = mlrun.new_project("proj1", save=False)
    default_image = "myrepo/myimage1"

    # Without a project default image, set_function with file-path in context and repo for remote kind must get an image
    with pytest.raises(ValueError, match="image must be provided"):
        proj1.set_source("git://mock.git", pull_at_runtime=False)
        # Specify the relative path for the file to be considered in the project's context
        proj1.set_function(
            func="./assets/handler.py",
            name="func",
            kind="job",
            handler="myhandler",
            with_repo=True,
        )

    proj1.set_default_image(default_image)
    proj1.set_function(func=func_path, name="func", kind="job", handler="myhandler")

    # Functions should remain without the default image in the project cache (i.e. without enrichment). Only with
    # enrichment, the default image should apply
    non_enriched_function = proj1.get_function("func", enrich=False)
    assert non_enriched_function.spec.image == ""
    enriched_function = proj1.get_function("func", enrich=True)
    assert enriched_function.spec.image == default_image

    # Same check - with a function object
    func1 = mlrun.code_to_function(
        "func2", kind="job", handler="myhandler", filename=func_path
    )
    proj1.set_function(func1, name="func2")

    non_enriched_function = proj1.get_function("func2", enrich=False)
    assert non_enriched_function.spec.image == ""
    enriched_function = proj1.get_function("func2", enrich=True)
    assert enriched_function.spec.image == default_image

    # If function already had an image, the default image must not override
    func1.spec.image = "some/other_image"
    proj1.set_function(func1, name="func3")

    enriched_function = proj1.get_function("func3", enrich=True)
    assert enriched_function.spec.image == "some/other_image"

    # Enrich the function in-place. Validate that changing the default image affects this function
    proj1.get_function("func", enrich=True, copy_function=False)
    new_default_image = "mynewrepo/mynewimage1"
    proj1.set_default_image(new_default_image)

    enriched_function = proj1.get_function("func")
    assert enriched_function.spec.image == new_default_image


def test_function_not_enriched_with_project_default_function_node_selector():
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    mlrun.mlconf.artifact_path = "/tmp"
    proj1 = mlrun.new_project("proj1", save=False)
    default_function_node_selector = {"gpu": "true"}

    non_enriched_function = proj1.set_function(
        func=func_path,
        name="func",
        kind="job",
        image="mlrun/mlrun",
        handler="myhandler",
    )
    assert non_enriched_function.spec.node_selector == {}

    proj1.default_function_node_selector = default_function_node_selector
    enriched_function = proj1.get_function("func", enrich=True)
    # Check that function is not affected by project
    assert (
        enriched_function.spec.node_selector == non_enriched_function.spec.node_selector
    )

    # Same check - with a function object
    func1 = mlrun.code_to_function(
        "func2",
        kind="job",
        handler="myhandler",
        image="mlrun/mlrun",
        filename=func_path,
    )
    proj1.set_function(func1, name="func2")

    non_enriched_function = proj1.get_function("func2", enrich=False)
    assert non_enriched_function.spec.node_selector == {}

    enriched_function = proj1.get_function("func2", enrich=True)
    assert enriched_function.spec.node_selector == {}

    # If a function already has a node selector defined, the project-level node selector should merge with it,
    # but only apply the merged node selector to the job object. The function itself should remain unaffected.
    func1.spec.node_selector = {"zone": "us-west"}
    proj1.set_function(func1, name="func3")

    enriched_function = proj1.get_function("func3", enrich=True)
    assert enriched_function.spec.node_selector == {"zone": "us-west"}


def test_project_exports_default_image():
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    default_image = "myrepo/myimage1"
    project = mlrun.new_project("proj1", save=False)
    project.set_default_image(default_image)

    project.export(str(project_file_path))
    imported_project = mlrun.load_project(
        "./", str(project_file_path), save=False, allow_cross_project=True
    )
    assert imported_project.default_image == default_image


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

    mlrun.pipeline_context.clear(with_project=True)
    # expected to call `get_project`
    mlrun.get_run_db().store_project("proj1", proj1)
    run6 = mlrun.run_function(proj1.get_function("f1"), project_object=proj1)
    assert run6.spec.output_path == proj1.spec.artifact_path


@pytest.mark.parametrize(
    "workflow_path,exception",
    [
        (
            "./",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match=str(
                    re.escape(
                        "Invalid 'workflow_path': './'. Got a path to a non-existing file. Path must be absolute or "
                        "relative to the project code path i.e. <project.spec.get_code_path()>/<workflow_path>)."
                    )
                ),
            ),
        ),
        (
            "https://test",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match=str(
                    re.escape(
                        "Invalid 'workflow_path': 'https://test'. Got a remote URL without a file suffix."
                    )
                ),
            ),
        ),
        (
            "",
            pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match=str(re.escape("workflow_path must be provided.")),
            ),
        ),
        ("https://test.py", does_not_raise()),
        # relative path
        ("./workflow.py", does_not_raise()),
        ("./assets/handler.py", does_not_raise()),
        # only file name
        ("workflow.py", does_not_raise()),
        ("assets/handler.py", does_not_raise()),
        # absolute path
        (
            str(pathlib.Path(__file__).parent / "assets" / "handler.py"),
            does_not_raise(),
        ),
    ],
)
def test_set_workflow_path_validation(chdir_to_test_location, workflow_path, exception):
    proj = mlrun.new_project("proj", save=False)
    with exception:
        proj.set_workflow("main", workflow_path)


def test_set_workflow_local_engine():
    proj = mlrun.new_project("proj", save=False)
    with pytest.raises(ValueError):
        proj.set_workflow("main", "workflow.py", schedule="*/5 * * * *", engine="local")


def test_run_non_existing_workflow(rundb_mock):
    proj = mlrun.new_project("proj", save=False)
    proj.set_function("hub://describe", "describe")
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        proj.run("non-existing-workflow")


def test_project_ops():
    # verify that project ops (run_function, ..) will use the right project (and not the pipeline_context)
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    proj1 = mlrun.new_project("proj1", save=False)
    proj1.set_function(func_path, "f1", image="mlrun/mlrun", handler="myhandler")

    proj2 = mlrun.new_project("proj2", save=False)
    proj2.set_function(func_path, "f2", image="mlrun/mlrun", handler="myhandler")

    run = proj1.run_function("f1", params={"x": 1}, local=True)
    assert run.spec.function.startswith("proj1/f1")
    assert run.output("y") == 2  # = x * 2

    run = proj2.run_function("f2", params={"x": 2}, local=True)
    assert run.spec.function.startswith("proj2/f2")
    assert run.output("y") == 4  # = x * 2


@pytest.mark.parametrize(
    "parameters,hyperparameters,expectation,run_saved",
    [
        (
            {"x": 2**63},
            None,
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            False,
        ),
        (
            {"x": -(2**63)},
            None,
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            False,
        ),
        ({"x": 2**63 - 1}, None, does_not_raise(), True),
        ({"x": -(2**63) + 1}, None, does_not_raise(), True),
        (
            None,
            {"x": [1, 2**63]},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            False,
        ),
        (
            None,
            {"x": [1, -(2**63)]},
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            False,
        ),
        (None, {"x": [3, 2**63 - 1]}, does_not_raise(), True),
        (None, {"x": [3, -(2**63) + 1]}, does_not_raise(), True),
    ],
)
def test_validating_large_int_params(
    rundb_mock, parameters, hyperparameters, expectation, run_saved
):
    func_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    proj1 = mlrun.new_project("proj1", save=False)
    proj1.set_function(func_path, "f1", image="mlrun/mlrun", handler="myhandler")

    rundb_mock.reset()
    with expectation:
        proj1.run_function(
            "f1",
            params=parameters,
            hyperparams=hyperparameters,
            local=True,
        )

    assert run_saved == (rundb_mock._runs != {})


def test_load_project_with_git_enrichment(
    context,
    rundb_mock,
):
    url = "git://github.com/mlrun/project-demo.git"
    project = mlrun.load_project(
        context=str(context), url=url, save=True, allow_cross_project=True
    )

    assert (
        project.spec.source == "git://github.com/mlrun/project-demo.git#refs/heads/main"
    )


def test_remove_owner_name_in_load_project_from_yaml():
    # Create project and generate owner name
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.spec.owner = "some_owner"

    # Load the project from yaml and validate that the owner name was removed
    project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    project.export(str(project_file_path))
    imported_project = mlrun.load_project(
        "./", str(project_file_path), save=False, allow_cross_project=True
    )
    assert project.spec.owner == "some_owner"
    assert imported_project.spec.owner is None


def test_set_secrets_file_not_found():
    # Create project and generate owner name
    project_name = "project-name"
    file_name = ".env-test"
    project = mlrun.new_project(project_name, save=False)
    with pytest.raises(mlrun.errors.MLRunNotFoundError) as excinfo:
        project.set_secrets(file_path=file_name)
    assert f"{file_name} does not exist" in str(excinfo.value)


def test_authenticated_git_action_with_remote_cleanup(mock_git_repo):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.spec.repo = mock_git_repo

    dummy = unittest.mock.Mock()
    project._run_authenticated_git_action(
        action=dummy, remote="origin", secrets={"GIT_TOKEN": "my-token"}
    )

    expected_calls = [
        unittest.mock.call(
            "https://my-token:x-oauth-basic@git.server/my-repo",
            "https://git.server/my-repo",
        ),
        unittest.mock.call(
            "https://git.server/my-repo",
            "https://my-token:x-oauth-basic@git.server/my-repo",
        ),
    ]

    dummy.assert_called_once()
    project.spec.repo.remotes["origin"].set_url.assert_has_calls(
        expected_calls,
        any_order=False,
    )
    project.spec.repo.remotes["organization"].set_url.assert_not_called()


def test_unauthenticated_git_action_with_remote_pristine(mock_git_repo):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    project.spec.repo = mock_git_repo

    dummy = unittest.mock.Mock()
    project._run_authenticated_git_action(
        action=dummy,
        remote="organization",
    )

    dummy.assert_called_once()
    project.spec.repo.remotes["organization"].set_url.assert_not_called()
    project.spec.repo.remotes["origin"].set_url.assert_not_called()


def test_get_or_create_project_no_db():
    mlrun.mlconf.dbpath = ""
    project_name = "project-name"
    project = mlrun.get_or_create_project(project_name, allow_cross_project=True)
    assert project.name == project_name


@pytest.mark.parametrize(
    "requirements ,with_requirements_file, commands",
    [
        (["pandas", "scikit-learn"], False, ["echo 123"]),
        (["pandas", "scikit-learn"], True, ["echo 123"]),
        ([], True, ["echo 123"]),
        (None, True, ["echo 123"]),
        ([], False, ["echo 123"]),
    ],
)
def test_project_build_config(requirements, with_requirements_file, commands):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    image = "my-image"
    requirements_file = str(assets_path() / "requirements-test.txt")
    project.build_config(
        image=image,
        requirements=requirements,
        requirements_file=requirements_file if with_requirements_file else None,
        commands=commands,
    )

    expected_requirements = requirements
    if with_requirements_file:
        expected_requirements = [
            "faker",
            "python-dotenv",
            "chardet>=3.0.2, <4.0",
        ] + (requirements or [])
    assert project.spec.build.image == image
    assert project.spec.build.requirements == expected_requirements
    assert project.spec.build.commands == commands


@pytest.mark.parametrize(
    "requirements ,with_requirements_file",
    [
        (["pandas", "scikit-learn"], False),
        (["pandas", "scikit-learn"], True),
        ([], True),
        (None, True),
        ([], False),
    ],
)
def test_project_set_function_with_requirements(requirements, with_requirements_file):
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    image = "my-image"
    requirements_file = str(assets_path() / "requirements-test.txt")
    func = project.set_function(
        name="my-func",
        image=image,
        func=str(assets_path() / "handler.py"),
        requirements=requirements,
        requirements_file=requirements_file if with_requirements_file else None,
    )

    expected_requirements = requirements
    if with_requirements_file:
        expected_requirements = [
            "faker",
            "python-dotenv",
            "chardet>=3.0.2, <4.0",
        ] + (requirements or [])

    if requirements or with_requirements_file:
        assert func.spec.build.base_image == image
    else:
        assert func.spec.image == image

    assert func.spec.build.requirements == expected_requirements

    # set from object
    if requirements or with_requirements_file:
        # change requirements to make sure they are overriden
        func.spec.build.requirements = ["some-req"]
    project.set_function(
        name="my-func",
        image=image,
        func=func,
        requirements=requirements,
        requirements_file=requirements_file if with_requirements_file else None,
    )

    if requirements or with_requirements_file:
        assert func.spec.build.base_image == image
    else:
        assert func.spec.image == image

    assert func.spec.build.requirements == expected_requirements


def test_init_function_from_dict_function_in_spec():
    project_name = "project-name"
    project = mlrun.new_project(project_name, save=False)
    func_dict = {
        "name": "sparkjob-from-github",
        "spec": {
            "kind": "spark",
            "metadata": {
                "name": "sparkjob-from-github",
                "tag": "latest",
                "project": project_name,
            },
            "spec": {
                "command": "simple_job.py",
                "image": ".sparkjob-from-github:latest",
                "build": {
                    "source": "./",
                    "base_image": "iguazio/spark-app:3.5.5-b697",
                    "load_source_on_run": False,
                    "requirements": ["pyspark==3.2.3"],
                    "source_code_target_dir": "/home/mlrun_code/",
                },
                "description": "",
                "disable_auto_mount": False,
                "clone_target_dir": "/home/mlrun_code/",
                "replicas": 1,
                "image_pull_policy": "Always",
                "priority_class_name": "dummy-class",
                "preemption_mode": "prevent",
                "driver_resources": {
                    "requests": {"memory": "512m", "cpu": 1},
                    "limits": {"cpu": "1300m"},
                },
                "executor_resources": {
                    "requests": {"memory": "512m", "cpu": 1},
                    "limits": {"cpu": "1400m"},
                },
                "deps": {
                    "jars": [
                        "local:///spark/v3io-libs/v3io-hcfs_2.12.jar",
                        "local:///spark/v3io-libs/v3io-spark3-streaming_2.12.jar",
                        "local:///spark/v3io-libs/v3io-spark3-object-dataframe_2.12.jar",
                        "local:///igz/java/libs/scala-library-2.12.14.jar",
                        "local:///spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
                    ],
                    "files": ["local:///igz/java/libs/v3io-pyspark.zip"],
                },
                "use_default_image": False,
                "monitoring": {
                    "enabled": True,
                    "exporter_jar": "/spark/jars/jmx_prometheus_javaagent-0.16.1.jar",
                },
                "driver_preemption_mode": "prevent",
                "executor_preemption_mode": "prevent",
                "affinity": None,
                "tolerations": None,
                "node_selector": None,
                "executor_affinity": None,
                "executor_tolerations": None,
                "executor_node_selector": None,
                "driver_affinity": None,
                "driver_tolerations": None,
                "driver_node_selector": None,
                "state_thresholds": mlrun.mlconf.function.spec.state_thresholds.default.to_dict(),
            },
            "verbose": False,
        },
    }
    func = mlrun.projects.project._init_function_from_dict(func_dict, project)
    assert (
        deepdiff.DeepDiff(func[1].to_dict(), func_dict["spec"], ignore_order=True) == {}
    )


def test_load_project_from_yaml_with_function(context):
    project_name = "project-name"
    project = mlrun.new_project(project_name, context=str(context), save=False)
    function = mlrun.code_to_function(
        name="my-func",
        image="my-image",
        kind="job",
        filename=str(assets_path() / "handler.py"),
    )
    function.save()
    project.set_function(function)
    project.set_function(
        name="my-other-func",
        image="my-image",
        func=str(assets_path() / "handler.py"),
        tag="latest",
    )
    project.save()
    loaded_project = mlrun.load_project(context=str(context), allow_cross_project=True)
    for function_name in ["my-func", "my-other-func"]:
        assert (
            deepdiff.DeepDiff(
                project.get_function(function_name).to_dict(),
                loaded_project.get_function(function_name).to_dict(),
                ignore_order=True,
                exclude_paths=[
                    "root['spec']['build']['code_origin']",
                ],
            )
            == {}
        )


@pytest.mark.parametrize(
    "kind_1 ,kind_2, canary, upstreams",
    [
        (
            "nuclio",
            "nuclio",
            [20, 80],
            [
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": "my-func1"}, percentage=80
                ),
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": "my-func2"}, percentage=20
                ),
            ],
        ),
        (
            "nuclio",
            None,
            None,
            [
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": "my-func1"}, percentage=0
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "authentication_mode",
    [
        mlrun.common.schemas.APIGatewayAuthenticationMode.none,
        mlrun.common.schemas.APIGatewayAuthenticationMode.basic,
        mlrun.common.schemas.APIGatewayAuthenticationMode.access_key,
    ],
)
@unittest.mock.patch.object(mlrun.db.nopdb.NopDB, "store_api_gateway")
def test_create_api_gateway_valid(
    patched_create_api_gateway,
    context,
    kind_1,
    kind_2,
    canary,
    upstreams,
    authentication_mode,
):
    mlrun.mlconf.igz_version = "3.6.0"
    patched_create_api_gateway.return_value = mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(
            name="new-gw",
            labels={
                mlrun_constants.MLRunInternalLabels.nuclio_project_name: "project-name"
            },
        ),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="new-gw",
            path="/",
            host="gateway-f1-f2-project-name.some-domain.com",
            upstreams=upstreams,
            authenticationMode=authentication_mode,
        ),
        status=mlrun.common.schemas.APIGatewayStatus(
            state=mlrun.common.schemas.APIGatewayState.ready,
        ),
    )
    project_name = "project-name"
    project = mlrun.new_project(project_name, context=str(context), save=False)
    f1 = mlrun.code_to_function(
        name="my-func1",
        image="my-image",
        kind=kind_1,
        filename=str(assets_path() / "handler.py"),
    )
    f1.save()
    functions = f1
    project.set_function(f1)
    if kind_2:
        f2 = mlrun.code_to_function(
            name="my-func2",
            image="my-image",
            kind=kind_2,
            filename=str(assets_path() / "handler.py"),
        )
        f2.save()
        project.set_function(f2)
        functions = [f1, f2]
    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
            name="gateway-f1-f2",
        ),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            functions=functions,
            canary=canary,
            project=project_name,
        ),
    )
    if authentication_mode == mlrun.common.schemas.APIGatewayAuthenticationMode.basic:
        api_gateway.with_basic_auth("test_username", "test_password")
    elif (
        authentication_mode
        == mlrun.common.schemas.APIGatewayAuthenticationMode.access_key
    ):
        api_gateway.with_access_key_auth()

    gateway = project.store_api_gateway(api_gateway=api_gateway)

    gateway_dict = gateway.to_dict()
    assert "metadata" in gateway_dict
    assert "spec" in gateway_dict

    assert gateway.invoke_url == "https://gateway-f1-f2-project-name.some-domain.com"
    if authentication_mode == mlrun.common.schemas.APIGatewayAuthenticationMode.basic:
        assert gateway.authentication.authentication_mode == "basicAuth"
    elif (
        authentication_mode
        == mlrun.common.schemas.APIGatewayAuthenticationMode.access_key
    ):
        assert gateway.authentication.authentication_mode == "accessKey"
    else:
        assert gateway.authentication.authentication_mode == "none"


@pytest.mark.parametrize(
    "kind_1 ,kind_2, canary",
    [
        ("nuclio", "nuclio", [20]),
        ("nuclio", "nuclio", [20, 10]),
        ("nuclio", "job", [20, 80]),
        ("job", None, None),
    ],
)
def test_create_api_gateway_invalid(context, kind_1, kind_2, canary):
    project_name = "project-name"
    project = mlrun.new_project(project_name, context=str(context), save=False)
    f1 = mlrun.code_to_function(
        name="my-func1",
        image="my-image",
        kind=kind_1,
        filename=str(assets_path() / "handler.py"),
    )
    f1.save()
    functions = f1
    project.set_function(f1)
    if kind_2:
        f2 = mlrun.code_to_function(
            name="my-func2",
            image="my-image",
            kind=kind_2,
            filename=str(assets_path() / "handler.py"),
        )
        f2.save()
        project.set_function(f2)
        functions = [f1, f2]
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.runtimes.nuclio.api_gateway.APIGateway(
            mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
                name="gateway-f1-f2",
            ),
            mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
                functions=functions,
                canary=canary,
                project=project_name,
            ),
        )


@unittest.mock.patch.object(mlrun.db.nopdb.NopDB, "list_api_gateways")
def test_list_api_gateways(patched_list_api_gateways, context):
    patched_list_api_gateways.return_value = mlrun.common.schemas.APIGatewaysOutput(
        api_gateways={
            "test": mlrun.common.schemas.APIGateway(
                metadata=mlrun.common.schemas.APIGatewayMetadata(
                    name="test",
                    labels={
                        mlrun_constants.MLRunInternalLabels.nuclio_project_name: "project-name"
                    },
                ),
                spec=mlrun.common.schemas.APIGatewaySpec(
                    name="test",
                    path="/",
                    host="http://gateway-f1-f2-project-name.some-domain.com",
                    upstreams=[
                        mlrun.common.schemas.APIGatewayUpstream(
                            nucliofunction={"name": "my-func1"}, percentage=0
                        ),
                    ],
                ),
            ),
            "test2": mlrun.common.schemas.APIGateway(
                metadata=mlrun.common.schemas.APIGatewayMetadata(
                    name="test2",
                    labels={
                        mlrun_constants.MLRunInternalLabels.nuclio_project_name: "project-name"
                    },
                ),
                spec=mlrun.common.schemas.APIGatewaySpec(
                    name="test2",
                    path="/",
                    host="http://test-basic-default.domain.com",
                    upstreams=[
                        mlrun.common.schemas.APIGatewayUpstream(
                            nucliofunction={"name": "my-func1"}, percentage=0
                        )
                    ],
                ),
            ),
        }
    )
    project_name = "project-name"
    project = mlrun.new_project(project_name, context=str(context), save=False)
    gateways = project.list_api_gateways()

    assert gateways[0].name == "test"
    assert gateways[0].host == "http://gateway-f1-f2-project-name.some-domain.com"
    assert gateways[0].spec.functions == ["project-name/my-func1"]

    assert gateways[1].invoke_url == "http://test-basic-default.domain.com"


def test_project_create_remote():
    # test calling create_remote without git_init=True on project creation

    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a project
        project_name = "project-name"
        project = mlrun.get_or_create_project(
            project_name, context=tmp_dir, allow_cross_project=True
        )

        project.create_remote(
            url="https://github.com/mlrun/some-git-repo.git",
            name="mlrun-remote",
        )

        assert project.spec.repo is not None
        assert "mlrun-remote" in [remote.name for remote in project.spec.repo.remotes]


@pytest.mark.parametrize(
    "url,set_url,name,set_name,overwrite,expected_url,expected",
    [
        # Remote doesn't exist, create normally
        (
            "https://github.com/mlrun/some-git-repo.git",
            "https://github.com/mlrun/some-other-git-repo.git",
            "mlrun-remote",
            "mlrun-another-remote",
            False,
            "https://github.com/mlrun/some-other-git-repo.git",
            does_not_raise(),
        ),
        # Remote exists, overwrite False, raise MLRunConflictError
        (
            "https://github.com/mlrun/some-git-repo.git",
            "https://github.com/mlrun/some-git-other-repo.git",
            "mlrun-remote",
            "mlrun-remote",
            False,
            "https://github.com/mlrun/some-git-repo.git",
            pytest.raises(mlrun.errors.MLRunConflictError),
        ),
        # Remote exists, overwrite True, update remote
        (
            "https://github.com/mlrun/some-git-repo.git",
            "https://github.com/mlrun/some-git-other-repo.git",
            "mlrun-remote",
            "mlrun-remote",
            True,
            "https://github.com/mlrun/some-git-other-repo.git",
            does_not_raise(),
        ),
    ],
)
def test_set_remote_as_update(
    url, set_url, name, set_name, overwrite, expected_url, expected
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a project
        project_name = "project-name"
        project = mlrun.get_or_create_project(
            project_name, context=tmp_dir, allow_cross_project=True
        )

        project.create_remote(
            url=url,
            name=name,
        )
        with expected:
            project.set_remote(
                url=set_url,
                name=set_name,
                overwrite=overwrite,
            )

            if name != set_name:
                assert project.spec.repo.remote(name).url == url
            assert project.spec.repo.remote(set_name).url == expected_url


@pytest.mark.parametrize(
    "url,name,expected",
    [
        # Remote doesn't exist, create normally
        (
            "https://github.com/mlrun/some-other-git-repo.git",
            "mlrun-remote2",
            does_not_raise(),
        ),
        # Remote exists, raise MLRunConflictError
        (
            "https://github.com/mlrun/some-git-repo.git",
            "mlrun-remote",
            pytest.raises(mlrun.errors.MLRunConflictError),
        ),
    ],
)
def test_create_remote(url, name, expected):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a project
        project_name = "project-name"
        project = mlrun.get_or_create_project(
            project_name, context=tmp_dir, allow_cross_project=True
        )

        project.create_remote(
            url="https://github.com/mlrun/some-git-repo.git",
            name="mlrun-remote",
        )

        with expected:
            project.create_remote(
                url=url,
                name=name,
            )
            assert project.spec.repo.remote(name).url == url


@pytest.mark.parametrize(
    "name",
    [
        # Remote exists
        "mlrun-remote",
        # Remote doesn't exist
        "non-existent-remote",
    ],
)
def test_remove_remote(name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a project
        project_name = "project-name"
        project = mlrun.get_or_create_project(
            project_name, context=tmp_dir, allow_cross_project=True
        )

        project.create_remote(
            url="https://github.com/mlrun/some-git-repo.git",
            name="mlrun-remote",
        )
        project.remove_remote(name)
        assert name not in project.spec.repo.remotes


@pytest.mark.parametrize(
    "source_url, pull_at_runtime, base_image, image_name, target_dir",
    [
        (None, None, "aaa/bbb", "ccc/ddd", None),
        ("git://some/repo", False, None, ".some-image", None),
        (
            "git://some/other/repo",
            False,
            ".some-base-image",
            "some-repo/some-target-image",
            "/target/path/for/source",
        ),
        ("git://some/repo", True, None, ".some-image", "/target/path"),
    ],
)
def test_project_build_image(
    source_url, pull_at_runtime, base_image, image_name, target_dir, remote_builder_mock
):
    project_name = "project1"

    project = mlrun.new_project(project_name, save=False)

    if source_url:
        project.set_source(source_url, pull_at_runtime=pull_at_runtime)

    project.build_image(image=image_name, base_image=base_image, target_dir=target_dir)

    (
        build_config,
        clone_target_dir,
    ) = remote_builder_mock.get_build_config_and_target_dir()

    # If pull-at-runtime, then source will not be provided to the build process since no configuration is needed
    # at build time. Also, there will be no clone_target_dir, since no pulling/cloning is happening at build.
    if pull_at_runtime:
        assert build_config.load_source_on_run is None
        assert build_config.source is None
        assert clone_target_dir is None
    else:
        assert not build_config.load_source_on_run
        assert build_config.source == source_url
        assert clone_target_dir == target_dir

    assert build_config.image == image_name
    # If no base image was used, then mlrun/mlrun is expected
    assert build_config.base_image == base_image or "mlrun/mlrun"
    assert project.default_image == image_name


@pytest.mark.parametrize(
    "project_name, valid",
    [
        ("project", True),
        ("project-name", True),
        ("project-name-1", True),
        ("1project", True),
        ("project_name", False),
        ("project@", False),
        ("project/a", False),
    ],
)
def test_project_name_validation(project_name, valid):
    assert valid == mlrun.projects.ProjectMetadata.validate_project_name(
        project_name, raise_on_failure=False
    )


@pytest.mark.parametrize(
    "project_labels, valid",
    [
        ({}, True),
        ({"key": "value"}, True),
        ({"some.key": "value"}, True),
        ({"key.some/a": "value"}, True),
        # too many subcomponents
        ({"key/a/b": "value"}, False),
        # must start with alphanumeric
        ({".key": "value"}, False),
        ({"/key": "value"}, False),
        # no key
        ({"": "value"}, False),
        # long value
        ({"key": "a" * 64}, False),
        # long key
        ({"a" * 64: "a"}, False),
    ],
)
def test_project_labels_validation(project_labels, valid):
    assert valid == mlrun.projects.ProjectMetadata.validate_project_labels(
        project_labels, raise_on_failure=False
    )


@pytest.mark.parametrize(
    "project_file_name, expectation",
    [
        ("project.yaml", does_not_raise()),
        ("project.yml", does_not_raise()),
        ("non-valid-file.yamrt", pytest.raises(mlrun.errors.MLRunNotFoundError)),
    ],
)
def test_load_project_dir(project_file_name, expectation):
    project_dir = "project-dir"
    os.makedirs(project_dir, exist_ok=True)
    try:
        # copy project.yaml from assets to project_dir
        shutil.copy(
            str(assets_path() / "project.yaml"),
            str(pathlib.Path(project_dir) / project_file_name),
        )
        with expectation:
            project = mlrun.load_project(
                project_dir, save=False, allow_cross_project=True
            )
            # just to make sure the project was loaded correctly from the file
            assert project.name == "pipe2"
    finally:
        shutil.rmtree(project_dir)


def test_workflow_path_with_project_workdir():
    project_name = "project1"

    project = mlrun.new_project(project_name, save=False, context="./context")
    workflow_path = "workflow.py"
    workflow_spec = mlrun.projects.pipelines.WorkflowSpec(path=workflow_path)
    # with_out_workdir
    path = workflow_spec.get_source_file(project.spec.get_code_path())
    assert path == "./context/workflow.py"

    # with__subpath
    project = mlrun.new_project(
        project_name, save=False, context="./context", subpath="./subpath"
    )
    path = workflow_spec.get_source_file(project.spec.get_code_path())
    assert path == "./context/./subpath/workflow.py"

    # with__workdir
    project.spec.workdir = "./workdir"
    path = workflow_spec.get_source_file(project.spec.get_code_path())
    assert path == "./context/./workdir/workflow.py"


@pytest.mark.parametrize(
    "alert_data",
    [None, ""],
)
def test_store_alert_config_missing_alert_data(alert_data):
    project_name = "dummy-project"
    project = mlrun.new_project(project_name, save=False)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="Alert data must be provided"
    ):
        project.store_alert_config(alert_data=alert_data)


def test_run_project_sync_functions_fails_silently(rundb_mock):
    proj = mlrun.new_project("proj", save=False)
    proj.spec._function_definitions = {
        "prep-data": {
            "url": "prep_data.py",
            "image": "mlrun/mlrun",
            "handler": "prep_data",
        },
        "train": {
            "url": "/User/some-notebook.ipynb",  # Absolute path
            "name": "train",
            "kind": "job",
            "image": "mlrun/mlrun",
            "handler": "trainer",
        },
    }
    name = "my-pipeline"
    proj.set_workflow(
        name=name,
        workflow_path=str(assets_path() / "localpipe.py"),
        handler="my_pipe",
    )

    # Sync should fail silently and run should fail as the functions were not saved
    run_status = proj.run(name)
    assert run_status.state == RunStatuses.failed
    assert "Function tstfunc not found" in str(run_status.exc)


class TestModelMonitoring:
    """Test model monitoring project methods"""

    @staticmethod
    @pytest.fixture
    def project() -> mlrun.projects.MlrunProject:
        return unittest.mock.Mock()

    @staticmethod
    def test_enable_wait_for_deployment(project: mlrun.projects.MlrunProject) -> None:
        with unittest.mock.patch.object(
            project, "_wait_for_functions_deployment", autospec=True
        ) as mock:
            mlrun.projects.MlrunProject.enable_model_monitoring(
                project, deploy_histogram_data_drift_app=False, wait_for_deployment=True
            )

        mock.assert_called_once()
        assert (
            mock.call_args_list[0].args[0] == mm_consts.MonitoringFunctionNames.list()
        ), "Expected to wait for the infra functions"
