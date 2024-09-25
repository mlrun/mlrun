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
import io
import os
import pathlib
import re
import shutil
import sys
import time
from sys import executable

import igz_mgmt
import mlrun_pipelines.common.models
import pandas as pd
import pytest
from kfp import dsl

import mlrun
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.utils.logger
import tests.system.common.helpers.notifications as notification_helpers
from mlrun.artifacts import Artifact
from mlrun.common.runtimes.constants import RunStates
from mlrun.model import EntrypointParam
from tests.conftest import out_path
from tests.system.base import TestMLRunSystem

data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"
model_class = "sklearn.linear_model.LogisticRegression"
projects_dir = f"{out_path}/proj"
funcs = mlrun.projects.pipeline_context.functions


def exec_project(args):
    cmd = [executable, "-m", "mlrun", "project"] + args
    out = os.popen(" ".join(cmd)).read()
    return out


# pipeline for inline test (run pipeline from handler)
@dsl.pipeline(name="test pipeline", description="test")
def pipe_test():
    # train the model using a library (hub://) function and the generated data
    funcs["auto-trainer"].as_step(
        name="train",
        inputs={"dataset": data_url},
        params={"model_class": model_class, "label_columns": "label"},
        outputs=["model", "test_set"],
    )


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestProject(TestMLRunSystem):
    project_name = "project-system-test-project"
    _logger_redirected = False

    def custom_setup(self):
        super().custom_setup()
        self.custom_project_names_to_delete = []

    def custom_teardown(self):
        super().custom_teardown()
        if self._logger_redirected:
            mlrun.utils.logger.replace_handler_stream("default", sys.stdout)
            self._logger_redirected = False

        self._logger.debug(
            "Deleting custom projects",
            num_projects_to_delete=len(self.custom_project_names_to_delete),
        )
        for name in self.custom_project_names_to_delete:
            self._delete_test_project(name)

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(self, project_name, with_repo=False, overwrite=False):
        self.custom_project_names_to_delete.append(project_name)
        self._logger.debug(
            "Creating new project",
            project_name=project_name,
            with_repo=False,
            overwrite=overwrite,
        )
        proj = mlrun.new_project(
            project_name, str(self.assets_path), overwrite=overwrite
        )
        proj.set_function(
            "prep_data.py",
            "prep-data",
            image="mlrun/mlrun",
            handler="prep_data",
            with_repo=with_repo,
        )
        proj.set_function("hub://describe")
        proj.set_function("hub://auto-trainer", "auto-trainer")
        proj.set_function("hub://v2-model-server", "serving")
        proj.set_artifact("data", Artifact(target_path=data_url))
        proj.spec.params = {"label_columns": "label"}
        arg = EntrypointParam(
            "model_class",
            type="str",
            default=model_class,
            doc="model package/algorithm",
        )
        proj.set_workflow("main", "./kflow.py", args_schema=[arg])
        proj.set_workflow("newflow", "./newflow.py", handler="newpipe")
        proj.spec.artifact_path = "v3io:///projects/{{run.project}}"
        self._logger.debug(
            "Saving project",
            project_name=project_name,
            project=proj.to_yaml(),
        )
        proj.save()
        return proj

    def test_project_persists_function_changes(self):
        func_name = "build-func"
        commands = [
            "echo 1111",
            "echo 2222",
        ]
        self.project.set_function(
            str(self.assets_path / "handler.py"),
            func_name,
            kind="job",
            image="mlrun/mlrun",
        )
        self.project.build_function(
            func_name, base_image="mlrun/mlrun", commands=commands
        )
        assert (
            self.project.get_function(func_name, sync=False).spec.build.commands
            == commands
        )

    def test_build_function_image_usability(self):
        func_name = "my-func"
        fn = self.project.set_function(
            str(self.assets_path / "handler.py"),
            func_name,
            kind="job",
            image="mlrun/mlrun",
        )

        # redirect logger to capture logs and check for warnings
        self._logger_redirected = True
        _stdout = io.StringIO()
        mlrun.utils.logger.replace_handler_stream("default", _stdout)

        # build function with image that has a protocol prefix
        self.project.build_function(
            fn,
            image=f"https://{mlrun.mlconf.httpdb.builder.docker_registry}/test/image:v3",
            base_image="mlrun/mlrun",
            commands=["echo 1"],
        )
        out = _stdout.getvalue()
        assert (
            "[mlrun:warning] The image has an unexpected protocol prefix ('http://' or 'https://'). "
            "If you wish to use the default configured registry, no protocol prefix is required "
            "(note that you can also use '.<image-name>' instead of the full URL "
            "where <image-name> is a placeholder). "
            "Removing protocol prefix from image." in out
        )

    def test_run(self):
        name = "pipe0"
        self._create_project(name)

        # load project from context dir and run a workflow
        project2 = mlrun.load_project(
            str(self.assets_path), name=name, allow_cross_project=True
        )
        run = project2.run("main", watch=True, artifact_path=f"v3io:///projects/{name}")
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

        # test the list_runs/artifacts/functions methods
        runs_list = project2.list_runs(name="test", labels=f"workflow={run.run_id}")
        runs = runs_list.to_objects()
        assert runs[0].status.state == "completed"
        assert runs[0].metadata.name == "test"
        runs_list.compare(filename=f"{projects_dir}/compare.html")
        artifacts = project2.list_artifacts(tree=run.run_id).to_objects()

        # model, prep_data_cleaned_data, test_evaluation-confusion-matrix, test_evaluation-roc-curves,
        # test_evaluation-test_set, train_confusion-matrix, train_feature-importance, train_roc-curves, test_set
        assert len(artifacts) == 9
        assert artifacts[0].producer["workflow"] == run.run_id

        models = project2.list_models(tree=run.run_id)
        assert len(models) == 1
        assert models[0].producer["workflow"] == run.run_id

        functions = project2.list_functions(tag="latest")
        assert len(functions) == 2  # prep-data, auto-trainer
        assert functions[0].metadata.project == name

    def test_run_artifact_path(self):
        name = "pipe1"
        self._create_project(name)

        # load project from context dir and run a workflow
        project = mlrun.load_project(
            str(self.assets_path), name=name, allow_cross_project=True
        )
        # Don't provide an artifact-path, to verify that the run-id is added by default
        workflow_run = project.run("main", watch=True)
        assert (
            workflow_run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

        # check that the functions running in the workflow had the output_path set correctly
        db = mlrun.get_run_db()
        run_id = workflow_run.run_id
        pipeline = db.get_pipeline(run_id, project=name)
        for graph_step in pipeline["graph"].values():
            if "run_uid" in graph_step:
                run_object = db.read_run(uid=graph_step["run_uid"], project=name)
                output_path = run_object["spec"]["output_path"]
                assert output_path == f"v3io:///projects/{name}/{run_id}"

    def test_run_git_load(self):
        # load project from git
        name = "pipe2"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        project2 = mlrun.load_project(
            project_dir,
            "git://github.com/mlrun/project-demo.git#main",
            name=name,
            allow_cross_project=True,
        )
        self._logger.info("run pipeline from git")

        # run project, load source into container at runtime
        project2.spec.load_source_on_run = True
        run = project2.run("main", artifact_path=f"v3io:///projects/{name}")
        run.wait_for_completion()
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

    def test_run_git_build(self):
        name = "pipe3"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # load project from git, build the container image from source (in the workflow)
        project2 = mlrun.load_project(
            project_dir,
            "git://github.com/mlrun/project-demo.git#main",
            name=name,
            allow_cross_project=True,
        )
        self._logger.info("run pipeline from git")
        project2.spec.load_source_on_run = False
        run = project2.run(
            "main",
            artifact_path=f"v3io:///projects/{name}",
            arguments={"build": 1},
        )
        run.wait_for_completion()
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

    @staticmethod
    def _assert_cli_output(output: str, project_name: str):
        m = re.search(" Pipeline run id=(.+),", output)
        assert m, "pipeline id is not in output"

        run_id = m.group(1).strip()
        db = mlrun.get_run_db()
        pipeline = db.get_pipeline(run_id, project=project_name)
        state = pipeline["run"]["status"]
        assert (
            state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

    def test_run_cli(self):
        # load project from git
        name = "pipe4"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # clone a project to local dir
        args = [
            "-n",
            name,
            "-u",
            "git://github.com/mlrun/project-demo.git",
            project_dir,
        ]
        out = exec_project(args)
        self._logger.debug("executed project", out=out)

        # load the project from local dir and change a workflow
        project2 = mlrun.load_project(project_dir, allow_cross_project=True)
        self.custom_project_names_to_delete.append(project2.metadata.name)
        project2.spec.workflows = {}
        project2.set_workflow("kf", "./kflow.py")
        project2.save()
        self._logger.debug("saved project", project2=project2.to_yaml())

        # exec the workflow
        args = [
            "-n",
            name,
            "-r",
            "kf",
            "-w",
            "-p",
            f"v3io:///projects/{name}",
            project_dir,
        ]
        out = exec_project(args)
        self._assert_cli_output(out, name)

    def test_cli_with_remote(self):
        # load project from git
        name = "pipe-remote-cli"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # clone a project to local dir
        args = [
            "--name",
            name,
            "--url",
            "git://github.com/mlrun/project-demo.git",
            project_dir,
        ]
        out = exec_project(args)
        self._logger.debug("Loaded project", out=out)

        # exec the workflow
        args = [
            "--name",
            name,
            "--run",
            "main",
            "--watch",
            "--engine",
            "remote",
            "--artifact-path",
            f"v3io:///projects/{name}",
            project_dir,
        ]
        out = exec_project(args)
        self._logger.debug("Executed project", out=out)

        assert re.search(
            "Workflow (.+) finished, state=Succeeded", out
        ), "workflow did not finished successfully"

    def test_inline_pipeline(self):
        name = "pipe5"
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = self._create_project(name, with_repo=True)
        run = project.run(
            artifact_path=f"v3io:///projects/{name}/artifacts",
            workflow_handler=pipe_test,
        )
        run.wait_for_completion()
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"

    def test_cli_no_save_flag(self):
        # load project from git
        name = "saveproj12345"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # clone a project to local dir but don't save the project to the DB
        args = [
            "-n",
            name,
            "-u",
            "git://github.com/mlrun/project-demo.git",
            "--no-save",
            project_dir,
        ]
        exec_project(args)

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self._run_db.get_project(name=name)

    def test_get_or_create(self):
        # create project and save to DB
        name = "newproj73"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.get_or_create_project(
            name, project_dir, allow_cross_project=True
        )
        project.spec.description = "mytest"
        project.save()

        # get project should read from DB
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.get_or_create_project(
            name, project_dir, allow_cross_project=True
        )
        project.save()
        assert project.spec.description == "mytest", "failed to get project"
        self._delete_test_project(name)

        # get project should read from context (project.yaml)
        project = mlrun.get_or_create_project(
            name, project_dir, allow_cross_project=True
        )
        assert project.spec.description == "mytest", "failed to get project"

    def test_new_project_overwrite(self):
        # create project and save to DB
        project_dir = f"{projects_dir}/{self.project_name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = self._create_project(self.project_name, overwrite=True)

        db = mlrun.get_run_db()
        project.sync_functions(save=True)
        project.register_artifacts()

        # get project from db for creation time
        project = db.get_project(name=self.project_name)

        assert len(project.list_functions()) == 4, "functions count mismatch"
        assert len(project.list_artifacts()) == 1, "artifacts count mismatch"
        old_creation_time = project.metadata.created

        project = mlrun.new_project(
            self.project_name, str(self.assets_path), overwrite=True
        )
        project.sync_functions(save=True)
        project.register_artifacts()
        project = db.get_project(name=self.project_name)

        assert (
            project.metadata.created > old_creation_time
        ), "creation time is not after overwritten project's creation time"

        # ensure cascading delete
        assert project.list_functions() is None, "project should not have functions"
        assert len(project.list_artifacts()) == 0, "artifacts count mismatch"

    def test_overwrite_project_failure(self):
        # create project and save to DB
        project_dir = f"{projects_dir}/{self.project_name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = self._create_project(self.project_name, overwrite=True)

        db = mlrun.get_run_db()
        project.sync_functions(save=True)
        project.register_artifacts()

        # get project from db for creation time
        project = db.get_project(name=self.project_name)

        assert len(project.list_functions()) == 4, "functions count mismatch"
        assert len(project.list_artifacts()) == 1, "artifacts count mismatch"
        old_creation_time = project.metadata.created

        # overwrite with invalid from_template value
        with pytest.raises(ValueError):
            mlrun.new_project(self.project_name, from_template="bla", overwrite=True)

        # ensure project was not deleted
        project = db.get_project(name=self.project_name)
        assert len(project.list_functions()) == 4, "functions count mismatch"
        assert len(project.list_artifacts()) == 1, "artifacts count mismatch"
        assert (
            project.metadata.created == old_creation_time
        ), "creation time was changed"

    def _test_new_pipeline(self, name, engine):
        project = self._create_project(name)
        project.set_function(
            "gen_iris.py",
            "gen-iris",
            image="mlrun/mlrun",
            handler="iris_generator",
            requirements=["requests"],
        )
        self._logger.debug("Set project function", project=project.to_yaml())
        run = project.run(
            "newflow",
            engine=engine,
            artifact_path=f"v3io:///projects/{name}",
            watch=True,
        )
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"
        fn = project.get_function("gen-iris", ignore_cache=True)
        assert fn.status.state == "ready"
        assert fn.spec.image, "image path got cleared"

    def test_local_pipeline(self):
        self._test_new_pipeline("lclpipe", engine="local")

    def test_kfp_pipeline(self):
        self._test_new_pipeline("kfppipe", engine="kfp")

    def test_kfp_runs_getting_deleted_on_project_deletion(self):
        project_name = "kfppipedelete"
        project = self._create_project(project_name)
        self._initialize_sleep_workflow(project)
        project.run("main", engine="kfp")

        db = mlrun.get_run_db()
        project_pipeline_runs = db.list_pipelines(project=project_name)
        self._logger.debug(
            "Got project pipeline runs", runs_length=len(project_pipeline_runs.runs)
        )
        # expecting to have pipeline run
        assert (
            project_pipeline_runs.runs
        ), "no pipeline runs found for project, expected to have pipeline run"
        # deleting project with deletion strategy cascade so it will delete any related resources ( pipelines as well )

        self._logger.debug("Deleting project", project_name=project_name)
        db.delete_project(
            name=project_name,
            deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascade,
        )
        # create the project again ( using new_project, instead of get_or_create_project so it won't create project
        # from project.yaml in the context that might contain project.yaml
        self._logger.debug("Recreating project", project_name=project_name)
        mlrun.new_project(project_name)

        project_pipeline_runs = db.list_pipelines(project=project_name)
        self._logger.debug(
            "Got project pipeline runs", runs_length=len(project_pipeline_runs.runs)
        )
        assert (
            not project_pipeline_runs.runs
        ), "pipeline runs found for project after deletion, expected to be empty"

    def test_kfp_pipeline_with_resource_param_passed(self):
        project_name = "test-pipeline-with-resource-param"
        self.custom_project_names_to_delete.append(project_name)
        project = mlrun.new_project(project_name, context=str(self.assets_path))

        code_path = str(self.assets_path / "sleep.py")
        workflow_path = str(self.assets_path / "pipeline_with_resource_param.py")

        project.set_function(
            name="func-1",
            func=code_path,
            kind="job",
            image="mlrun/mlrun",
            handler="handler",
        )
        # set and run a two-step workflow in the project
        project.set_workflow("paramflow", workflow_path)

        arguments = {"memory": "11Mi"}
        pipeline_status = project.run(
            "paramflow", engine="kfp", arguments=arguments, watch=True
        )
        assert pipeline_status.workflow.args == arguments

        # get the function from the db
        function = project.get_function("func-1", ignore_cache=True)
        assert function.spec.resources["requests"]["memory"] == arguments["memory"]

    def test_remote_pipeline_with_workflow_runner_node_selector(self):
        project_name = "rmtpipe-kfp-github"
        self.custom_project_names_to_delete.append(project_name)

        workflow_name = "newflow"
        workflow_runner_name = f"workflow-runner-{workflow_name}"
        runner_node_selector = {"kubernetes.io/arch": "amd64"}
        project_default_function_node_selector = {"kubernetes.io/os": "linux"}

        project = self._load_remote_pipeline_project(name=project_name)
        project.spec.default_function_node_selector = (
            project_default_function_node_selector
        )
        project.save()

        project.run(
            workflow_name,
            engine="remote",
            workflow_runner_node_selector=runner_node_selector,
        )
        runner_run_result = project.list_runs(name=workflow_runner_name)[0]
        assert runner_run_result["spec"]["node_selector"] == {
            **project_default_function_node_selector,
            **runner_node_selector,
        }
        # The workflow execution includes a load_project, which clears the node_selector from the project.
        # As a result, we need to reapply the node_selector
        project.spec.default_function_node_selector = (
            project_default_function_node_selector
        )
        project.save()

        # Test scheduled workflow
        schedule = "0 0 30 2 *"
        project.run(
            workflow_name,
            engine="remote",
            workflow_runner_node_selector=runner_node_selector,
            schedule=schedule,
        )

        # Invoke the workflow manually
        mlrun.get_run_db().invoke_schedule(project=project_name, name=workflow_name)
        runner_run_result = project.list_runs(labels="job-type=workflow-runner")[0]
        assert runner_run_result["spec"]["node_selector"] == {
            **project_default_function_node_selector,
            **runner_node_selector,
        }

    def test_remote_pipeline_with_kfp_engine_from_github(self):
        project_name = "rmtpipe-kfp-github"
        self.custom_project_names_to_delete.append(project_name)

        self._test_remote_pipeline_from_github(
            name=project_name,
            workflow_name="newflow",
            engine="remote",
            watch=True,
            notification_steps={
                # gen data function build step
                "build": 1,
                # gen data, summary, train and test steps
                "run": 4,
                # serving step
                "deploy": 1,
            },
        )
        self._test_remote_pipeline_from_github(
            name=project_name, workflow_name="main", engine="remote:kfp"
        )

    def test_remote_pipeline_with_local_engine_from_github(self):
        project_name = "rmtpipe-local-github"
        self.custom_project_names_to_delete.append(project_name)

        self._test_remote_pipeline_from_github(
            name=project_name,
            workflow_name="newflow",
            engine="remote:local",
            watch=True,
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            self._test_remote_pipeline_from_github(
                name=project_name,
                workflow_name="newflow",
                engine="remote",
                local=True,
            )

    def test_non_existent_run_id_in_pipeline(self):
        project_name = "default"
        db = mlrun.get_run_db()

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.get_pipeline(
                "25811259-6d21-4caf-86e8-badc0ffee000", project=project_name
            )

    def test_remote_from_archive(self):
        name = "pipe6"
        project = self._create_project(name)
        archive_path = f"v3io:///projects/{project.name}/archive1.zip"
        project.export(archive_path)
        project.spec.source = archive_path
        project.save()
        self._logger.debug("Saved project", project=project.to_yaml())
        run = project.run(
            "main",
            watch=True,
            engine="remote",
        )
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"
        assert run.run_id, "workflow's run id failed to fetch"

    def test_kfp_from_local_code(self):
        name = "kfp-from-local-code"
        self.custom_project_names_to_delete.append(name)

        # change cwd to the current file's dir to make sure the handler file is found
        current_file_abspath = os.path.abspath(__file__)
        current_dirname = os.path.dirname(current_file_abspath)
        os.chdir(current_dirname)

        project = mlrun.get_or_create_project(
            name, user_project=True, context="./", allow_cross_project=True
        )

        handler_fn = project.set_function(
            func="./assets/handler.py",
            handler="my_func",
            name="my-func",
            kind="job",
            image="mlrun/mlrun",
        )
        project.build_function(handler_fn)

        project.set_workflow(
            "main", "./assets/handler_workflow.py", handler="job_pipeline"
        )
        project.save()

        run = project.run(
            "main",
            watch=True,
        )
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"
        assert run.run_id, "workflow's run id failed to fetch"

    def test_local_cli(self):
        # load project from git
        name = "lclclipipe"
        project = self._create_project(name)
        project.set_function(
            "gen_iris.py",
            "gen-iris",
            image="mlrun/mlrun",
            handler="iris_generator",
        )
        project.save()
        self._logger.debug("saved project", project=project.to_yaml())

        # exec the workflow
        args = [
            "-n",
            name,
            "-r",
            "newflow",
            "--engine",
            "local",
            "-w",
            "-p",
            f"v3io:///projects/{name}",
            str(self.assets_path),
        ]
        out = exec_project(args)
        self._logger.debug("executed project", out=out)
        assert (
            out.find("Pipeline run finished, state=Succeeded") != -1
        ), "pipeline failed"

    def test_run_cli_watch_with_timeout(self):
        name = "run-cli-watch-with-timeout"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # exec the workflow and set a short timeout, should fail
        args = [
            "--name",
            name,
            "--url",
            "git://github.com/mlrun/project-demo.git",
            "--run",
            "main",
            "--watch",
            "--timeout 1",
            project_dir,
        ]
        out = exec_project(args)

        self._logger.debug("executed project", out=out)
        assert (
            out.find(
                "Failed to execute command by the given deadline. "
                "last_exception: pipeline run has not completed yet, "
                "function_name: _wait_for_pipeline_completion, timeout: 1, "
                "caused by: pipeline run has not completed yet"
            )
            != -1
        )

    def test_build_and_run(self):
        # test that build creates a proper image and run will use the updated function (with the built image)
        name = "buildandrun"
        self.custom_project_names_to_delete.append(name)
        project = mlrun.new_project(name, context=str(self.assets_path))

        # test with user provided function object
        base_fn = mlrun.code_to_function(
            "scores",
            filename=str(self.assets_path / "sentiment.py"),
            kind="job",
            image="mlrun/mlrun",
            requirements=["vaderSentiment"],
            handler="handler",
        )

        fn = base_fn.copy()
        assert fn.spec.build.base_image == "mlrun/mlrun" and not fn.spec.image
        fn.spec.build.with_mlrun = False
        project.build_function(fn)
        run_result = project.run_function(fn, params={"text": "good morning"})
        assert run_result.output("score")

        # test with function from project spec
        project.set_function(
            "./sentiment.py",
            "scores2",
            kind="job",
            image="mlrun/mlrun",
            requirements=["vaderSentiment"],
            handler="handler",
        )
        project.build_function("scores2")
        run_result = project.run_function("scores2", params={"text": "good morning"})
        assert run_result.output("score")

        # test auto build option (the function will be built on the first time, then run)
        fn = base_fn.copy()
        fn.metadata.name = "scores3"
        fn.spec.build.auto_build = True
        run_result = project.run_function(fn, params={"text": "good morning"})
        assert fn.status.state == "ready"
        assert fn.spec.image, "image path got cleared"
        assert run_result.output("score")

        # Use project default image to run function, don't specify image when calling set_function
        project.set_default_image(fn.spec.image)
        project.set_function(
            "./sentiment.py",
            "scores4",
            kind="job",
            handler="handler",
        )
        enriched_fn = project.get_function("scores4", enrich=True)
        assert enriched_fn.spec.image == fn.spec.image
        project.run_function("scores4", params={"text": "good evening"})
        assert fn.status.state == "ready"
        assert run_result.output("score")

    def test_run_function_uses_user_defined_artifact_path_over_project_artifact_path(
        self,
    ):
        func_name = "my-func"
        self.project.set_function(
            str(self.assets_path / "handler.py"),
            func_name,
            kind="job",
            image="mlrun/mlrun",
        )

        self.project.artifact_path = "/User/project_artifact_path"
        user_artifact_path = "/User/user_artifact_path"

        run = self.project.run_function(func_name, artifact_path=user_artifact_path)
        assert run.spec.output_path == user_artifact_path

    def test_set_secrets(self):
        name = "set-secrets"
        self.custom_project_names_to_delete.append(name)
        project = mlrun.new_project(name, context=str(self.assets_path))
        project.save()
        env_file = str(self.assets_path / "envfile")
        db = mlrun.get_run_db()
        db.delete_project_secrets(name, provider="kubernetes")
        project.set_secrets(file_path=env_file)
        secrets = db.list_project_secret_keys(name, provider="kubernetes")
        assert secrets.secret_keys == ["ENV_ARG1", "ENV_ARG2"]

    def test_override_workflow(self):
        name = "override-test"
        project_dir = f"{projects_dir}/{name}"
        workflow_name = "main"
        self.custom_project_names_to_delete.append(name)
        project = mlrun.load_project(
            project_dir,
            "git://github.com/mlrun/project-demo.git",
            name=name,
            allow_cross_project=True,
        )

        schedules = ["*/30 * * * *", "*/40 * * * *", "*/50 * * * *"]
        # overwriting nothing
        project.run(workflow_name, schedule=schedules[0])
        schedule = self._run_db.get_schedule(name, workflow_name)
        assert (
            schedule.scheduled_object["schedule"] == schedules[0]
        ), "Failed to override nothing"

        # overwriting schedule:
        project.run(workflow_name, schedule=schedules[1], dirty=True)
        schedule = self._run_db.get_schedule(name, workflow_name)
        assert (
            schedule.scheduled_object["schedule"] == schedules[1]
        ), "Failed to override existing workflow"

        # overwriting schedule from cli:
        args = [
            project_dir,
            "-n",
            name,
            "-d",
            "-r",
            workflow_name,
            "--schedule",
            f"'{schedules[2]}'",
        ]
        exec_project(args)
        schedule = self._run_db.get_schedule(name, workflow_name)
        assert (
            schedule.scheduled_object["schedule"] == schedules[2]
        ), "Failed to override from CLI"

    def test_timeout_warning(self):
        name = "timeout-warning-test"
        project_dir = f"{projects_dir}/{name}"
        workflow_name = "main"
        bad_timeout = "6"
        good_timeout = "12"
        self.custom_project_names_to_delete.append(name)
        mlrun.load_project(
            project_dir,
            "git://github.com/mlrun/project-demo.git",
            name=name,
            allow_cross_project=True,
        )

        args = [
            project_dir,
            "-n",
            name,
            "-d",
            "-r",
            workflow_name,
            "--timeout",
            bad_timeout,
        ]
        out = exec_project(args)
        warning_message = (
            "[warning] Timeout ({}) must be higher than backoff (10)."
            " Set timeout to be higher than backoff."
        )
        expected_warning_log = warning_message.format(bad_timeout)
        assert expected_warning_log in out

        args = [
            project_dir,
            "-n",
            name,
            "-d",
            "-r",
            workflow_name,
            "--timeout",
            good_timeout,
        ]
        out = exec_project(args)
        unexpected_warning_log = warning_message.format(good_timeout)
        assert unexpected_warning_log not in out

    def test_failed_schedule_workflow_non_remote_source(self):
        name = "non-remote-fail"
        # Creating a local project
        project = self._create_project(name)

        # scheduling project with non-remote source (scheduling)
        run = project.run("main", schedule="*/10 * * * *")
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.failed
        ), f"pipeline should failed, state = {run.state}"

        # scheduling project with non-remote source (single run)
        run = project.run("main", engine="remote")
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.failed
        ), f"pipeline should failed, state = {run.state}"

    def test_remote_workflow_source(self):
        name = "source-project"
        project_dir = f"{projects_dir}/{name}"
        original_source = "git://github.com/mlrun/project-demo.git"
        temporary_source = original_source + "#yaronha-patch-1"
        self.custom_project_names_to_delete.append(name)
        artifact_path = f"v3io:///projects/{name}"

        project = mlrun.load_project(
            project_dir,
            original_source,
            name=name,
            allow_cross_project=True,
        )
        # Getting the expected source after possible enrichment:
        expected_source = project.source

        run = project.run(
            "newflow",
            engine="local",
            local=True,
            source=temporary_source,
            artifact_path=artifact_path,
        )
        assert run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        # Ensuring that the project's source has not changed in the db:
        project_from_db = self._run_db.get_project(name)
        assert project_from_db.source == expected_source

        for engine in ["remote", "local", "kfp"]:
            project.run(
                "main",
                engine=engine,
                source=temporary_source,
                artifact_path=artifact_path,
                dirty=True,
            )

            # Ensuring that the project's source has not changed in the db:
            project_from_db = self._run_db.get_project(name)
            assert project_from_db.source == expected_source

        # Ensuring that the loaded project is from the given source
        run = project.run(
            "newflow",
            engine="remote",
            source=temporary_source,
            dirty=True,
            artifact_path=artifact_path,
            watch=True,
        )
        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.failed
        ), "pipeline supposed to fail since newflow is not in the temporary source"

    def test_workflow_image_fails(self):
        name = "test-image"
        self.custom_project_names_to_delete.append(name)

        # _create_project contains set_workflow inside:
        project = self._create_project(name)
        new_workflow = project.workflows[0]
        new_workflow["image"] = "not-existed"
        new_workflow["name"] = "bad-image"
        project.spec.workflows = project.spec.workflows + [new_workflow]

        archive_path = f"v3io:///projects/{project.name}/archived.zip"
        project.export(archive_path)
        project.spec.source = archive_path
        project.save()

        run = project.run(
            "bad-image",
            engine="remote",
        )
        assert run.state == mlrun_pipelines.common.models.RunStatuses.failed

    def _assert_scheduled(self, project_name, schedule_str):
        schedule = self._run_db.get_schedule(project_name, "main")
        assert schedule.scheduled_object["schedule"] == schedule_str

    def test_remote_workflow_source_with_subpath(self):
        # Test running remote workflow when the project files are store in a relative path (the subpath)
        project_source = "git://github.com/mlrun/system-tests.git#main"
        project_context = "./test_subpath_remote"
        project_name = "test-remote-workflow-source-with-subpath"
        self.custom_project_names_to_delete.append(project_name)
        project = mlrun.load_project(
            context=project_context,
            url=project_source,
            subpath="./test_remote_workflow_subpath",
            name=project_name,
            allow_cross_project=True,
        )
        project.save()
        project.run("main", arguments={"x": 1}, engine="remote:kfp", watch=True)

    @pytest.mark.parametrize("pull_state_mode", ["disabled", "enabled"])
    def test_abort_step_in_workflow(self, pull_state_mode):
        project_name = "test-abort-step"
        self.custom_project_names_to_delete.append(project_name)
        project = mlrun.new_project(project_name, context=str(self.assets_path))

        # when pull_state mode is enabled it simulates the flow of wait_for_completion
        mlrun.mlconf.httpdb.logs.pipelines.pull_state.mode = pull_state_mode

        def _assert_workflow_status(workflow, status):
            assert workflow.state == status

        self._initialize_sleep_workflow(project)

        # run a two-step workflow in the project
        workflow = project.run("main", engine="kfp")

        mlrun.utils.retry_until_successful(
            1,
            20,
            self._logger,
            True,
            _assert_workflow_status,
            workflow,
            mlrun_pipelines.common.models.RunStatuses.running,
        )

        # obtain the first run in the workflow when it began running
        runs = []
        while len(runs) != 1:
            runs = project.list_runs(
                labels=[f"workflow={workflow.run_id}"], state="running"
            )

        # abort the first workflow step
        db = mlrun.get_run_db()
        db.abort_run(runs.to_objects()[0].uid())

        # when a step is aborted, assert that the entire workflow failed and did not continue
        mlrun.utils.retry_until_successful(
            5,
            60,
            self._logger,
            True,
            _assert_workflow_status,
            workflow,
            mlrun_pipelines.common.models.RunStatuses.failed,
        )

    def _create_and_validate_project_function_with_node_selector(
        self, project: mlrun.projects.MlrunProject
    ):
        function_name = "test-func"
        function_label_name, function_label_val = "kubernetes.io/os", "linux"
        function_override_label, function_override_val = "kubernetes.io/hostname", ""

        code_path = str(self.assets_path / "sleep.py")
        func = project.set_function(
            name=function_name,
            func=code_path,
            kind="job",
            image="mlrun/mlrun",
            handler="handler",
        )
        func.spec.node_selector = {
            function_label_name: function_label_val,
            function_override_label: function_override_val,
        }

        # We run the function to ensure node selector enrichment, which doesn't occur during function build,
        # but at runtime.
        job = project.run_function(function_name)

        # Verify that the node selector is correctly enriched on job object
        assert job.spec.node_selector == {
            **mlrun.mlconf.get_default_function_node_selector(),
            **project.spec.default_function_node_selector,
            function_override_label: function_override_val,
            function_label_name: function_label_val,
        }

        # Verify that the node selector is not enriched on function object
        result_func = project.get_function(function_name)
        assert result_func.spec.node_selector == {
            function_label_name: function_label_val,
            function_override_label: function_override_val,
        }

    def _create_and_validate_mpi_function_with_node_selector(
        self, project: mlrun.projects.MlrunProject
    ):
        function_name = "test-func"
        function_label_name, function_label_val = "kubernetes.io/os", "linux"
        function_override_label, function_override_val = "kubernetes.io/hostname", ""

        code_path = str(self.assets_path / "mpijob_function.py")
        replicas = 2

        mpi_func = mlrun.code_to_function(
            name=function_name,
            kind="mpijob",
            handler="handler",
            project=project.name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        mpi_func.spec.replicas = replicas
        mpi_func.spec.node_selector = {
            function_label_name: function_label_val,
            function_override_label: function_override_val,
        }
        # We run the function to ensure node selector enrichment, which doesn't occur during function build,
        # but at runtime.
        mpijob_run = mpi_func.run(returns=["reduced_result", "rank_0_result"])
        assert mpijob_run.status.state == RunStates.completed

        # Verify that the node selector is correctly enriched on job object
        assert mpijob_run.spec.node_selector == {
            **mlrun.mlconf.get_default_function_node_selector(),
            **project.spec.default_function_node_selector,
            function_override_label: function_override_val,
            function_label_name: function_label_val,
        }

    @pytest.mark.enterprise
    def test_project_default_function_node_selector_using_igz_mgmt(self):
        project_label_name, project_label_val = "kubernetes.io/arch", "amd64"

        # Test using Iguazio to create the project
        project_name = "test-project"
        self.custom_project_names_to_delete.append(project_name)

        igz_mgmt.Project.create(
            self._igz_mgmt_client,
            name=project_name,
            owner="admin",
            default_function_node_selector=[
                {"name": project_label_name, "value": project_label_val}
            ],
        )

        project = self._run_db.get_project(project_name)
        assert project.spec.default_function_node_selector == {
            project_label_name: project_label_val
        }
        self._create_and_validate_project_function_with_node_selector(project)

    def _create_and_validate_spark_function_with_project_node_selectors(self, project):
        function_name = "spark-function"
        function_label_name, function_label_val = "kubernetes.io/os", "linux"
        function_override_label, function_override_val = "kubernetes.io/hostname", ""
        file_name = "spark.py"

        self._files_to_upload.append(file_name)
        self._upload_code_to_cluster()
        code_path = os.path.join(self.remote_code_dir, file_name)
        spark_function = mlrun.new_function(
            name=function_name,
            kind="spark",
            command=code_path.replace("v3io:///", "/v3io/"),
        )
        spark_function.with_igz_spark()
        spark_function.with_driver_limits(cpu="1300m")
        spark_function.with_driver_requests(cpu=1, mem="512m")

        spark_function.with_executor_limits(cpu="1400m")
        spark_function.with_executor_requests(cpu=1, mem="512m")

        node_selector = {
            function_label_name: function_label_val,
            function_override_label: function_override_val,
        }

        spark_function.with_node_selection(node_selector=node_selector)

        assert spark_function.spec.driver_node_selector == node_selector
        assert spark_function.spec.executor_node_selector == node_selector

        spark_function.with_igz_spark()

        spark_run = spark_function.run(auto_build=True)
        assert spark_run.status.state == RunStates.completed

    def test_project_default_function_node_selector(self):
        project_label_name, project_label_val = "kubernetes.io/arch", "amd64"
        project_label_to_remove, project_label_to_remove_val = (
            "kubernetes.io/hostname",
            "k8s-node1",
        )

        # Test using mlrun sdk to create the project
        project_name = "test-project"
        self.custom_project_names_to_delete.append(project_name)

        project = mlrun.new_project(
            project_name,
            default_function_node_selector={
                project_label_name: project_label_val,
                project_label_to_remove: project_label_to_remove_val,
            },
        )
        assert project.spec.default_function_node_selector == {
            project_label_name: project_label_val,
            project_label_to_remove: project_label_to_remove_val,
        }

        # self._create_and_validate_project_function_with_node_selector(project)
        # self._create_and_validate_mpi_function_with_node_selector(project)
        self._create_and_validate_spark_function_with_project_node_selectors(project)

    def test_project_build_image(self):
        name = "test-build-image"
        self.custom_project_names_to_delete.append(name)
        project = mlrun.new_project(name, context=str(self.assets_path))

        image_name = ".test-custom-image"
        project.build_image(
            image=image_name,
            set_as_default=True,
            with_mlrun=False,
            base_image="mlrun/mlrun",
            requirements=["vaderSentiment"],
            commands=["echo 1"],
        )

        assert project.default_image == image_name

        # test with user provided function object
        project.set_function(
            str(self.assets_path / "sentiment.py"),
            name="scores",
            kind="job",
            handler="handler",
        )

        run_result = project.run_function("scores", params={"text": "good morning"})
        assert run_result.output("score")

    def test_project_build_config_export_import(self):
        # Verify that the build config is exported properly by the project, and a new project loaded from it
        # can build default image directly without needing additional details.

        name_export = "test-build-image-export"
        name_import = "test-build-image-import"
        self.custom_project_names_to_delete.extend([name_export, name_import])

        project = mlrun.new_project(name_export, context=str(self.assets_path))
        image_name = ".test-export-custom-image"

        project.build_config(
            image=image_name,
            set_as_default=True,
            with_mlrun=False,
            base_image="mlrun/mlrun",
            requirements=["vaderSentiment"],
            commands=["echo 1"],
        )
        assert project.default_image == image_name

        project_dir = f"{projects_dir}/{name_export}"
        proj_file_path = project_dir + "/project.yaml"
        project.export(proj_file_path)

        new_project = mlrun.load_project(
            project_dir, name=name_import, allow_cross_project=True
        )
        new_project.build_image()

        new_project.set_function(
            str(self.assets_path / "sentiment.py"),
            name="scores",
            kind="job",
            handler="handler",
        )

        run_result = new_project.run_function(
            "scores", params={"text": "terrible evening"}
        )
        assert run_result.output("score")

        shutil.rmtree(project_dir, ignore_errors=True)

    def test_export_import_dataset_artifact(self):
        project_1_name = "project-1"
        self.custom_project_names_to_delete.append(project_1_name)
        project_1 = mlrun.new_project(project_1_name, context=str(self.assets_path))

        # create a dataset artifact
        local_path = f"{str(self.assets_path)}/my-df.parquet"
        data = {"col1": [1, 2], "col2": [3, 4]}
        data_frame = pd.DataFrame(data=data)
        key = "my-df"
        data_frame.to_parquet(local_path)
        dataset_artifact = mlrun.artifacts.dataset.DatasetArtifact(
            key, df=data_frame, format="parquet", target_path=local_path
        )
        project_1.log_artifact(dataset_artifact)

        # export the artifact to a zip file
        dataset_artifact = project_1.get_artifact(key)
        export_path = f"{str(self.assets_path)}/exported_dataset.zip"
        dataset_artifact.export(export_path)

        # create a new project and import the artifact
        project_2_name = "project-2"
        self.custom_project_names_to_delete.append(project_2_name)
        project_2 = mlrun.new_project(project_2_name, context=str(self.assets_path))

        imported_artifact = project_2.import_artifact(export_path)
        imported_artifact.to_dict()

        # validate that the artifact was imported properly and was uploaded to the store
        data_item = mlrun.get_dataitem(imported_artifact.target_path).get()
        assert data_item

    def test_export_import_zip_artifact(self):
        project_1_name = "project-1"
        self.custom_project_names_to_delete.append(project_1_name)
        project_1 = mlrun.new_project(project_1_name, context=str(self.assets_path))

        # create a file artifact that will be zipped by the packager
        create_artifact_function = project_1.set_function(
            func="create_file_artifact.py",
            name="create-artifact",
            kind="job",
            image="mlrun/mlrun",
        )
        create_artifact_function.run(
            handler="create_file_artifact",
            local=True,
            returns=["text_dir: path"],
        )

        # export the artifact to a zip file
        artifact = project_1.get_artifact(
            "create-artifact-create-file-artifact_text_dir"
        )
        exported_path = os.path.join(self.assets_path, "artifact.zip")
        artifact.export(exported_path)

        # create a new project and import the artifact
        project_2_name = "project-2"
        self.custom_project_names_to_delete.append(project_2_name)
        project_2 = mlrun.new_project(project_2_name, context=str(self.assets_path))

        new_artifact_key = "new-artifact"
        project_2.import_artifact(exported_path, new_key=new_artifact_key)

        use_artifact_function = project_2.set_function(
            func="use_artifact.py", name="use-artifact", kind="job", image="mlrun/mlrun"
        )

        # try to use the artifact in a function
        use_artifact_run = use_artifact_function.run(
            handler="use_artifact",
            local=True,
            inputs={"artifact": project_2.get_artifact_uri(new_artifact_key)},
        )

        # make sure the function run was successful, meaning the artifact was extracted successfully
        # from the zip by the packager
        assert (
            use_artifact_run.state()
            not in mlrun.common.runtimes.constants.RunStates.error_states()
        )

        exported_artifact = project_2.get_artifact(new_artifact_key)
        assert exported_artifact.target_path == artifact.target_path

    def test_load_project_with_artifact_db_key(self):
        project_1_name = "test-load-with-artifact"
        project_2_name = project_1_name + "-2"
        project_3_name = project_1_name + "-3"
        self.custom_project_names_to_delete.extend(
            [project_1_name, project_2_name, project_3_name]
        )

        context = "./load"
        project = mlrun.get_or_create_project(
            project_1_name, context=context, allow_cross_project=True
        )

        # create artifact with an explicit db_key
        artifact_key = "artifact_key"
        artifact_db_key = "artifact_db_key"
        project.log_artifact(artifact_key, db_key=artifact_db_key, body="test")

        # validate that the artifact is in the db
        artifacts = project.list_artifacts(name=artifact_db_key)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key

        # set the artifact on the project with a new key and save
        artifact_new_key = "artifact_new_key"
        project.set_artifact(
            key=artifact_new_key, artifact=Artifact.from_dict(artifacts[0])
        )
        project.save()

        # create a project from the same spec
        project2 = mlrun.load_project(
            context=context, name=project_2_name, allow_cross_project=True
        )

        # validate that the artifact was saved with the db_key
        artifacts = project2.list_artifacts(name=artifact_db_key)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_new_key

        # create another artifact with an explicit db_key
        artifact_db_key_2 = f"{artifact_db_key}_2"
        artifact_key_2 = f"{artifact_key}_2"
        project.log_artifact(
            f"{artifact_key_2}", db_key=artifact_db_key_2, body="test-again"
        )

        artifacts = project.list_artifacts(name=artifact_db_key_2)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == artifact_key_2

        # export the artifact and set it on the project with the export path
        artifact_path = os.path.join(os.getcwd(), context, "test-artifact.yaml")
        art = Artifact.from_dict(artifacts[0])
        art.export(artifact_path)
        another_artifact_key = "another_artifact_key"
        project.set_artifact(another_artifact_key, artifact=artifact_path)
        project.save()

        # create a new project from the same spec, and validate the artifact was loaded properly
        project3 = mlrun.load_project(
            context=context, name=project_3_name, allow_cross_project=True
        )
        # since it is imported from yaml, the artifact is saved with the set key
        artifacts = project3.list_artifacts(name=another_artifact_key)
        assert len(artifacts) == 1
        assert artifacts[0]["metadata"]["key"] == another_artifact_key

    def _initialize_sleep_workflow(self, project: mlrun.projects.MlrunProject):
        code_path = str(self.assets_path / "sleep.py")
        workflow_path = str(self.assets_path / "workflow.py")
        project.set_function(
            name="func-1",
            func=code_path,
            kind="job",
            image="mlrun/mlrun",
            handler="handler",
        )
        project.set_function(
            name="func-2",
            func=code_path,
            kind="job",
            image="mlrun/mlrun",
            handler="handler",
        )
        self._logger.debug("Set project workflow", project=project.name)
        project.set_workflow("main", workflow_path)

    @pytest.mark.parametrize(
        "name, save_secrets",
        [
            (
                "load-project-secrets",
                True,
            ),
            (
                "load-project-secrets-1",
                False,
            ),
        ],
    )
    def test_load_project_remotely_with_secrets(
        self,
        name,
        save_secrets,
    ):
        self.custom_project_names_to_delete.append(name)
        db = self._run_db
        state = db.load_project(
            name=name,
            url="git://github.com/mlrun/project-demo.git",
            secrets={"secret1": "1234"},
            save_secrets=save_secrets,
        )
        assert state == "completed"

        secrets = db.list_project_secret_keys(name)

        if save_secrets:
            assert "secret1" in secrets.secret_keys
        else:
            assert "secret1" not in secrets.secret_keys

    def test_load_project_remotely_with_secrets_failed(self):
        name = "failed-to-load"
        self.custom_project_names_to_delete.append(name)
        db = self._run_db
        state = db.load_project(
            name=name,
            url="git://github.com/some/wrong/uri.git",
            secrets={"secret1": "1234"},
            save_secrets=False,
        )
        assert state == "error"
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.get_project(name)

    def test_remote_workflow_source_on_image(self):
        name = "pipe"
        self.custom_project_names_to_delete.append(name)

        project_dir = f"{projects_dir}/{name}"
        source = "git://github.com/mlrun/project-demo.git"
        source_code_target_dir = (
            "./project"  # Optional, results to /home/mlrun_code/project
        )
        artifact_path = f"v3io:///projects/{name}"

        project = mlrun.load_project(
            project_dir,
            source,
            name=name,
            allow_cross_project=True,
        )
        project.set_source(source)

        # Build the image, load the source to the target dir and save the project
        project.build_image(target_dir=source_code_target_dir)
        project.save()

        run = project.run(
            "main",
            engine="remote",
            source="./",  # Relative to project.spec.build.source_code_target_dir
            artifact_path=artifact_path,
            dirty=True,
            watch=True,
        )
        assert run.state == mlrun.run.RunStatuses.succeeded

        # Ensuring that the project's source has not changed in the db:
        project_from_db = self._run_db.get_project(name)
        assert project_from_db.source == source

    @staticmethod
    def _generate_pipeline_notifications(
        nuclio_function_url: str,
    ) -> list[mlrun.model.Notification]:
        return [
            mlrun.model.Notification.from_dict(
                {
                    "kind": "webhook",
                    "name": "Pipeline Completed",
                    "message": "Pipeline Completed",
                    "severity": "info",
                    "when": ["completed"],
                    "condition": "",
                    "params": {
                        "url": nuclio_function_url,
                    },
                    "secret_params": {
                        "webhook": "some-webhook",
                    },
                }
            ),
        ]

    @staticmethod
    def _assert_pipeline_notification_steps(
        nuclio_function_url: str, notification_steps: dict
    ):
        # in order to trigger the periodic monitor runs function, to detect the failed run and send an event on it
        time.sleep(35)

        notification_data = list(
            notification_helpers.get_notifications_from_nuclio_and_reset_notification_cache(
                nuclio_function_url
            )
        )[0]
        notification_data_steps = {}
        for step in notification_data:
            if not step.get("step_kind"):
                # If there is not step_kind in the step, it means that it is the workflow runner, so we skip it
                continue
            notification_data_steps.setdefault(step.get("step_kind"), 0)
            notification_data_steps[step.get("step_kind")] += 1

        assert notification_data_steps == notification_steps

    def _load_remote_pipeline_project(self, name):
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.load_project(
            project_dir,
            "git://github.com/mlrun/project-demo.git",
            name=name,
            allow_cross_project=True,
        )
        return project

    def _test_remote_pipeline_from_github(
        self,
        name,
        workflow_name,
        engine=None,
        local=None,
        watch=False,
        notification_steps=None,
    ):
        project = self._load_remote_pipeline_project(name=name)

        nuclio_function_url = None
        notifications = []
        if notification_steps:
            # nuclio function for storing notifications, to validate the notifications from the pipeline
            nuclio_function_url = notification_helpers.deploy_notification_nuclio(
                project
            )
            notifications = self._generate_pipeline_notifications(nuclio_function_url)

        run = project.run(
            workflow_name,
            watch=watch,
            local=local,
            engine=engine,
            notifications=notifications,
        )

        assert (
            run.state == mlrun_pipelines.common.models.RunStatuses.succeeded
        ), "pipeline failed"
        # run.run_id can be empty in case of a local engine:
        assert run.run_id is not None, "workflow's run id failed to fetch"

        if notification_steps:
            self._assert_pipeline_notification_steps(
                nuclio_function_url, notification_steps
            )
