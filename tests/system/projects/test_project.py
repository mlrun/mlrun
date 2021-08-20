import pathlib
import shutil
import sys

import pytest
from kfp import dsl

import mlrun
from mlrun.artifacts import Artifact
from mlrun.model import EntrypointParam
from mlrun.utils import logger
from tests.conftest import out_path
from tests.system.base import TestMLRunSystem

data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"
model_pkg_class = "sklearn.linear_model.LogisticRegression"
projects_dir = f"{out_path}/proj"


# pipeline for inline test (run pipeline from handler)
@dsl.pipeline(name="test pipeline", description="test")
def pipe_test():
    # train the model using a library (hub://) function and the generated data
    train = funcs["train"].as_step(
        name="train",
        inputs={"dataset": data_url},
        params={"model_pkg_class": model_pkg_class, "label_column": "label",},
        outputs=["model", "test_set"],
    )


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestFeatureStore(TestMLRunSystem):
    def custom_setup(self):
        pass

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(self, project_name, with_repo=False):
        proj = mlrun.new_project(project_name, str(self.assets_path))
        proj.set_function(
            "prep_data.py",
            "prep-data",
            image="mlrun/mlrun",
            handler="prep_data",
            with_repo=with_repo,
        )
        proj.set_function("hub://sklearn_classifier", "train")
        proj.set_function("hub://test_classifier", "test")
        proj.set_function("hub://v2_model_server", "serve")
        proj.set_artifact("data", Artifact(target_path=data_url))
        proj.spec.params = {"label_column": "label"}
        arg = EntrypointParam(
            "model_pkg_class",
            type="str",
            default=model_pkg_class,
            doc="model package/algorithm",
        )
        proj.set_workflow("main", "./kflow.py", args_schema=[arg])
        proj.save()
        return proj

    def test_run(self):
        name = "pipe1"
        # create project in context
        self._create_project(name)

        # load project from context dir and run a workflow
        project2 = mlrun.load_project(str(self.assets_path), name=name)
        run = project2.run("main", watch=True, artifact_path=f"v3io:///projects/{name}")
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    def test_run_git_load(self):
        # load project from git
        name = "pipe2"
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        project2 = mlrun.load_project(
            project_dir, "git://github.com/mlrun/project-demo.git#main", name=name
        )
        logger.info("run pipeline from git")

        # run project, load source into container at runtime
        project2.spec.load_source_on_run = True
        run = project2.run("main", artifact_path=f"v3io:///projects/{name}")
        run.wait_for_completion()
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    def test_run_git_build(self):
        name = "pipe3"
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)

        # load project from git, build the container image from source (in the workflow)
        project2 = mlrun.load_project(
            project_dir, "git://github.com/mlrun/project-demo.git#main", name=name
        )
        logger.info("run pipeline from git")
        project2.spec.load_source_on_run = False
        run = project2.run(
            "main",
            artifact_path=f"v3io:///projects/{name}",
            arguments={"build": 1},
            workflow_path=str(self.assets_path / "kflow.py"),
        )
        run.wait_for_completion()
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    def test_inline_pipeline(self):
        name = "pipe4"
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = self._create_project(name, True)
        run = project.run(
            artifact_path=f"v3io:///projects/{name}", workflow_handler=pipe_test,
        )
        run.wait_for_completion()
        assert run.state == mlrun.run.RunStatuses.succeeded, "pipeline failed"

    def test_get_or_create(self):
        # create project and save to DB
        name = "newproj73"
        project_dir = f"{projects_dir}/{name}"
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.get_or_create_project(name, project_dir)
        project.spec.description = "mytest"
        project.save()

        # get project should read from DB
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.get_or_create_project(name, project_dir)
        assert project.spec.description == "mytest", "failed to get project"
