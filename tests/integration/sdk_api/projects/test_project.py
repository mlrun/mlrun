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
#
import pathlib

import deepdiff
import pytest

import mlrun
import mlrun.common.schemas
import tests.conftest
import tests.integration.sdk_api.base


class TestProject(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_create_project(self):
        project_name = "some-project"
        mlrun.new_project(project_name)
        projects = mlrun.get_run_db().list_projects()
        assert len(projects) == 1
        assert projects[0] == project_name

    def test_create_project_failure_already_exists(self):
        project_name = "some-project"
        mlrun.new_project(project_name)

        with pytest.raises(mlrun.errors.MLRunConflictError) as exc:
            mlrun.new_project(project_name)
        assert (
            f"Project with name {project_name} already exists. Use overwrite=True to overwrite the existing project."
            in str(exc.value)
        )

    def test_sync_functions(self):
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

        fn = project.get_function("describe")
        assert fn.metadata.name == "describe", "func did not return"

        # test that functions can be fetched from the DB (w/o set_function)
        mlrun.import_function("hub://auto_trainer", new_name="train").save()
        fn = project.get_function("train")
        assert fn.metadata.name == "train", "train func did not return"

    def test_overwrite_project(self):
        project_name = "some-project"

        # verify overwrite does not fail if project does not exist
        project = mlrun.new_project(project_name, overwrite=True)
        db = mlrun.get_run_db()

        # create several functions with several tags
        labels = {
            "name": "value",
            "name2": "value2",
        }
        function = mlrun.runtimes.KubejobRuntime()
        for label_name, label_value in labels.items():
            function.set_label(label_name, label_value)

        function_names = ["function-name-1", "function-name-2", "function-name-3"]
        function_tags = ["some-tag", "some-tag2", "some-tag3"]
        for function_name in function_names:
            for function_tag in function_tags:
                db.store_function(
                    function.to_dict(),
                    function_name,
                    project.metadata.name,
                    tag=function_tag,
                    versioned=True,
                )

        # create several artifacts
        artifact = {
            "kind": "artifact",
            "metadata": {"labels": labels},
            "spec": {"src_path": "/some/path"},
            "status": {"bla": "blabla"},
        }
        artifact_keys = ["artifact_key_1", "artifact_key_2", "artifact_key_3"]
        for artifact_key in artifact_keys:
            db.store_artifact(
                artifact_key,
                artifact,
                "some_uid",
                tag="some-tag",
                project=project.metadata.name,
            )

        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

        # verify artifacts and functions were created
        project_artifacts = project.list_artifacts()
        loaded_project_artifacts = projects[0].list_artifacts()
        assert (
            len(project_artifacts) == len(artifact_keys) * 2
        )  # project artifacts include "latest"
        assert len(project_artifacts) == len(loaded_project_artifacts)
        assert project_artifacts == loaded_project_artifacts

        project_functions = project.list_functions()
        loaded_project_functions = projects[0].list_functions()
        assert len(project_functions) == len(function_names) * len(function_tags)
        assert len(project_functions) == len(loaded_project_functions)

        loaded_project_functions_dicts = [
            func.to_dict() for func in loaded_project_functions
        ]
        assert all(
            [
                func.to_dict() in loaded_project_functions_dicts
                for func in project_functions
            ]
        )

        old_creation_time = projects[0].metadata.created

        mlrun.new_project(project_name, overwrite=True)
        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

        # ensure cascade deletion strategy
        assert projects[0].list_artifacts() == []
        assert projects[0].list_functions() is None
        assert projects[0].metadata.created > old_creation_time

    def test_overwrite_empty_project(self):
        project_name = "some-project"

        mlrun.new_project(project_name)
        db = mlrun.get_run_db()

        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

        assert projects[0].list_artifacts() == []
        assert projects[0].list_functions() is None
        old_creation_time = projects[0].metadata.created

        # overwrite empty project
        mlrun.new_project(project_name, overwrite=True)
        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

        assert projects[0].list_artifacts() == []
        assert projects[0].list_functions() is None
        assert projects[0].metadata.created > old_creation_time

    def test_overwrite_project_failure(self):
        project_name = "some-project"

        mlrun.new_project(project_name)
        db = mlrun.get_run_db()

        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name
        old_creation_time = projects[0].metadata.created

        # overwrite with invalid from_template value
        with pytest.raises(ValueError):
            mlrun.new_project(project_name, from_template="bla", overwrite=True)

        # ensure project was not deleted
        projects = db.list_projects(format_=mlrun.common.schemas.ProjectsFormat.full)
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name
        assert projects[0].metadata.created == old_creation_time

    def test_load_project_from_db(self):
        project_name = "some-project"
        mlrun.new_project(project_name)
        mlrun.load_project(".", f"db://{project_name}")

    def test_load_project_with_save(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
        project.export(str(project_file_path))

        imported_project_name = "imported-project"
        # loaded project but didn't saved
        mlrun.load_project(
            "./", str(project_file_path), name=imported_project_name, save=False
        )

        # loading project from db, but earlier load didn't saved, expected to fail
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            mlrun.load_project(".", f"db://{imported_project_name}", save=False)

        # loading project and saving
        expected_project = mlrun.load_project(
            "./", str(project_file_path), name=imported_project_name
        )

        # loading project from db, expected to succeed
        loaded_project_from_db = mlrun.load_project(
            ".", f"db://{imported_project_name}", save=False
        )
        _assert_projects(expected_project, loaded_project_from_db)

    def test_get_project(self):
        project_name = "some-project"
        # create an empty project
        mlrun.get_or_create_project(project_name)
        # get it from the db
        project = mlrun.get_or_create_project(project_name)

        # verify default values
        assert project.metadata.name == project_name
        assert project.metadata.labels == {}
        assert project.metadata.annotations == {}
        assert project.spec.params == {}
        assert project.spec.functions == []
        assert project.spec.workflows == []
        assert project.spec.artifacts == []
        assert project.spec.conda == ""

    def test_set_project_secrets(self):
        # A basic test verifying that we can access (mocked) project-secrets functionality in integration tests.
        project_name = "some-project"
        project_object = mlrun.get_or_create_project(project_name)

        secrets = {"secret1": "value1", "secret2": "value2"}
        project_object.set_secrets(secrets)
        secret_keys = (
            mlrun.get_run_db().list_project_secret_keys(project_name).secret_keys
        )
        assert secret_keys == list(secrets.keys())


def _assert_projects(expected_project, project):
    assert (
        deepdiff.DeepDiff(
            expected_project.to_dict(),
            project.to_dict(),
            ignore_order=True,
            exclude_paths={
                "root['metadata']['created']",
                "root['spec']['desired_state']",
                "root['status']",
            },
        )
        == {}
    )
    assert expected_project.spec.desired_state == project.spec.desired_state
    assert expected_project.spec.desired_state == project.status.state


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
