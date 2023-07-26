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
from sys import executable

import pytest
from kfp import dsl

import mlrun
import mlrun.common.schemas
import mlrun.utils.logger
from mlrun.artifacts import Artifact
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
    custom_project_names_to_delete = []
    _logger_redirected = False

    def custom_setup(self):
        pass

    def custom_teardown(self):
        if self._logger_redirected:
            mlrun.utils.logger.replace_handler_stream("default", sys.stdout)
            self._logger_redirected = False

        self._logger.debug(
            "Deleting custom projects",
            num_projects_to_delete=len(self.custom_project_names_to_delete),
        )
        for name in self.custom_project_names_to_delete:
            self._delete_test_project(name)

        self.custom_project_names_to_delete = []

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(self, project_name, with_repo=False, overwrite=False):
        self.custom_project_names_to_delete.append(project_name)
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
            image=f"https://{mlrun.config.config.httpdb.builder.docker_registry}/test/image:v3",
            base_image="mlrun/mlrun",
            commands=["echo 1"],
        )
        out = _stdout.getvalue()
        assert (
            "[warning] The image has an unexpected protocol prefix ('http://' or 'https://'). "
            "If you wish to use the default configured registry, no protocol prefix is required "
            "(note that you can also use '.<image-name>' instead of the full URL "
            "where <image-name> is a placeholder). "
            "Removing protocol prefix from image." in out
        )

    def test_run(self):
        name = "pipe0"
        self._create_project(name)

        # load project from context dir and run a workflow
        project2 = mlrun.load_project(str(self.assets_path), name=name)
        run = project2.run("main", watch=True, artifact_path=f"v3io:///projects/{name}")
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

        # test the list_runs/artifacts/functions methods
        runs_list = project2.list_runs(name="test", labels={"workflow": run.run_id})
        runs = runs_list.to_objects()
        assert runs[0].status.state == "completed"
        assert runs[0].metadata.name == "test"
        runs_list.compare(filename=f"{projects_dir}/compare.html")
        artifacts = project2.list_artifacts(tag=run.run_id).to_objects()

        # model, prep_data_cleaned_data, test_evaluation-confusion-matrix, test_evaluation-roc-curves,
        # test_evaluation-test_set, train_confusion-matrix, train_feature-importance, train_roc-curves, test_set
        assert len(artifacts) == 9
        assert artifacts[0].producer["workflow"] == run.run_id

        models = project2.list_models(tag=run.run_id)
        assert len(models) == 1
        assert models[0].producer["workflow"] == run.run_id

        functions = project2.list_functions(tag="latest")
        assert len(functions) == 2  # prep-data, auto-trainer
        assert functions[0].metadata.project == name

    def test_run_artifact_path(self):
        name = "pipe1"
        self._create_project(name)

        # load project from context dir and run a workflow
        project = mlrun.load_project(str(self.assets_path), name=name)
        # Don't provide an artifact-path, to verify that the run-id is added by default
        workflow_run = project.run("main", watch=True)
        assert workflow_run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

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
            project_dir, "git://github.com/mlrun/project-demo.git#main", name=name
        )
        self._logger.info("run pipeline from git")

        # run project, load source into container at runtime
        project2.spec.load_source_on_run = True
        run = project2.run("main", artifact_path=f"v3io:///projects/{name}")
        run.wait_for_completion()
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    def test_run_git_build(self):
        name = "pipe3"
        self.custom_project_names_to_delete.append(name)
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # load project from git, build the container image from source (in the workflow)
        project2 = mlrun.load_project(
            project_dir, "git://github.com/mlrun/project-demo.git#main", name=name
        )
        self._logger.info("run pipeline from git")
        project2.spec.load_source_on_run = False
        run = project2.run(
            "main",
            artifact_path=f"v3io:///projects/{name}",
            arguments={"build": 1},
        )
        run.wait_for_completion()
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    @staticmethod
    def _assert_cli_output(output: str, project_name: str):
        m = re.search(" Pipeline run id=(.+),", output)
        assert m, "pipeline id is not in output"

        run_id = m.group(1).strip()
        db = mlrun.get_run_db()
        pipeline = db.get_pipeline(run_id, project=project_name)
        state = pipeline["run"]["status"]
        assert state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

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
        project2 = mlrun.load_project(project_dir)
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
            "--ensure-project",
            project_dir,
        ]
        out = exec_project(args)
        self._assert_cli_output(out, name)

    def test_cli_with_remote(self):
        # load project from git
        name = "pipermtcli"
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

        # exec the workflow
        args = [
            "-n",
            name,
            "-r",
            "main",
            "-w",
            "--engine",
            "remote",
            "-p",
            f"v3io:///projects/{name}",
            project_dir,
        ]
        out = exec_project(args)
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
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

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
        project = mlrun.get_or_create_project(name, project_dir)
        project.spec.description = "mytest"
        project.save()

        # get project should read from DB
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.get_or_create_project(name, project_dir)
        project.save()
        assert project.spec.description == "mytest", "failed to get project"
        self._delete_test_project(name)

        # get project should read from context (project.yaml)
        project = mlrun.get_or_create_project(name, project_dir)
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
        self._logger.debug("set project function", project=project.to_yaml())
        run = project.run(
            "newflow",
            engine=engine,
            artifact_path=f"v3io:///projects/{name}",
            watch=True,
        )
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"
        fn = project.get_function("gen-iris", ignore_cache=True)
        assert fn.status.state == "ready"
        assert fn.spec.image, "image path got cleared"

    def test_local_pipeline(self):
        self._test_new_pipeline("lclpipe", engine="local")

    def test_kfp_pipeline(self):
        self._test_new_pipeline("kfppipe", engine="kfp")

    def test_kfp_runs_getting_deleted_on_project_deletion(self):
        project_name = "kfppipedelete"
        self.custom_project_names_to_delete.append(project_name)

        project = self._create_project(project_name)
        self._initialize_sleep_workflow(project)
        project.run("main", engine="kfp")

        db = mlrun.get_run_db()
        project_pipeline_runs = db.list_pipelines(project=project_name)
        # expecting to have pipeline run
        assert (
            project_pipeline_runs.runs
        ), "no pipeline runs found for project, expected to have pipeline run"
        # deleting project with deletion strategy cascade so it will delete any related resources ( pipelines as well )
        db.delete_project(
            name=project_name,
            deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascade,
        )
        # create the project again ( using new_project, instead of get_or_create_project so it won't create project
        # from project.yaml in the context that might contain project.yaml
        mlrun.new_project(project_name)

        project_pipeline_runs = db.list_pipelines(project=project_name)
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

    def _test_remote_pipeline_from_github(
        self, name, workflow_name, engine=None, local=None, watch=False
    ):
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.load_project(
            project_dir, "git://github.com/mlrun/project-demo.git", name=name
        )
        run = project.run(
            workflow_name,
            watch=watch,
            local=local,
            engine=engine,
        )

        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"
        # run.run_id can be empty in case of a local engine:
        assert run.run_id is not None, "workflow's run id failed to fetch"

    def test_remote_pipeline_with_kfp_engine_from_github(self):
        project_name = "rmtpipe-kfp-github"
        self.custom_project_names_to_delete.append(project_name)

        self._test_remote_pipeline_from_github(
            name=project_name,
            workflow_name="main",
            engine="remote",
            watch=True,
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
        self._logger.debug("saved project", project=project.to_yaml())
        run = project.run(
            "main",
            watch=True,
            engine="remote",
        )
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"
        assert run.run_id, "workflow's run id failed to fetch"

    def test_kfp_from_local_code(self):
        name = "kfp-from-local-code"
        self.custom_project_names_to_delete.append(name)

        # change cwd to the current file's dir to make sure the handler file is found
        current_file_abspath = os.path.abspath(__file__)
        current_dirname = os.path.dirname(current_file_abspath)
        os.chdir(current_dirname)

        project = mlrun.get_or_create_project(name, user_project=True, context="./")

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
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"
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
            out.find("pipeline run finished, state=Succeeded") != -1
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
                "failed to execute command by the given deadline. "
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
            "[warning] timeout ({}) must be higher than backoff (10)."
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
            run.state == mlrun.run.RunStatuses.failed
        ), f"pipeline should failed, state = {run.state}"

        # scheduling project with non-remote source (single run)
        run = project.run("main", engine="remote")
        assert (
            run.state == mlrun.run.RunStatuses.failed
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
        assert run.state == mlrun.run.RunStatuses.succeeded
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
            run.state == mlrun.run.RunStatuses.failed
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
        assert run.state == mlrun.run.RunStatuses.failed

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
            mlrun.run.RunStatuses.running,
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
            mlrun.run.RunStatuses.failed,
        )

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

        new_project = mlrun.load_project(project_dir, name=name_import)
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
        project.set_workflow("main", workflow_path)

    @pytest.mark.parametrize(
        "name, save_secrets, expected_states",
        [
            (
                "load-project-secrets",
                True,
                [
                    mlrun.common.schemas.BackgroundTaskState.running,
                    mlrun.common.schemas.BackgroundTaskState.succeeded,
                ],
            ),
            (
                "load-project-secrets-1",
                False,
                [mlrun.common.schemas.BackgroundTaskState.succeeded],
            ),
        ],
    )
    def test_load_project_remotely_with_secrets(
        self, name, save_secrets, expected_states
    ):
        self.custom_project_names_to_delete.append(name)
        db = self._run_db
        bg_task = db.load_project(
            name=name,
            url="git://github.com/mlrun/project-demo.git",
            secrets={"secret1": "1234"},
            save_secrets=save_secrets,
        )
        assert bg_task.status.state in expected_states

        secrets = db.list_project_secret_keys(name)
        if save_secrets:
            assert "secret1" in secrets.secret_keys
        else:
            assert "secret1" not in secrets.secret_keys

    def test_load_project_remotely_with_secrets_failed(self):
        name = "failed-to-load"
        db = self._run_db
        bg_task = db.load_project(
            name=name,
            url="git://github.com/some/wrong/uri.git",
            secrets={"secret1": "1234"},
            save_secrets=False,
        )
        assert bg_task.status.state == mlrun.common.schemas.BackgroundTaskState.failed
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.get_project(name)
