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
import os
import tempfile

import pytest

import mlrun
import tests.system.base
from mlrun.runtimes.constants import RunStates

git_uri = "git://github.com/mlrun/test-git-load.git"
base_image = "mlrun/mlrun"
tags = ["main", "refs/heads/tst"]
codepaths = [(None, "rootfn"), ("subdir", "func")]

job_cases = {
    # name: (command, workdir, handler, tag)
    "root-hndlr": ("", None, "rootfn.job_handler", tags[0]),
    "subdir-hndlr": ("", "subdir", "func.job_handler", tags[0]),
    "subdir-hndlr-ref": ("", "subdir", "func.job_handler", tags[1]),
    "root-cmd": ("rootfn.py", None, "job_handler", tags[1]),
}

# for private repo tests set the MLRUN_SYSTEM_TESTS_PRIVATE_REPO, MLRUN_SYSTEM_TESTS_PRIVATE_GIT_TOKEN env vars
private_repo = os.environ.get(
    "MLRUN_SYSTEM_TESTS_PRIVATE_REPO",
    "git://github.com/mlrun/private_git_tests.git#main",
)
has_private_source = (
    "MLRUN_SYSTEM_TESTS_PRIVATE_GIT_TOKEN" in os.environ and private_repo
)
need_private_git = pytest.mark.skipif(
    not has_private_source, reason="env vars for private git repo not set"
)


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestArchiveSources(tests.system.base.TestMLRunSystem):

    project_name = "git-tests"
    custom_project_names_to_delete = []

    def custom_setup(self):
        self.remote_code_dir = f"v3io:///projects/{self.project_name}/code/"
        self.uploaded_code = False
        # upload test files to cluster
        if has_private_source:
            self.project.set_secrets(
                {
                    "GIT_TOKEN": os.environ["MLRUN_SYSTEM_TESTS_PRIVATE_GIT_TOKEN"],
                }
            )

    def custom_teardown(self):
        for name in self.custom_project_names_to_delete:
            self._delete_test_project(name)

    def _upload_code_to_cluster(self):
        if not self.uploaded_code:
            for file in [
                "source_archive.tar.gz",
                "source_archive.zip",
                "handler.py",
                "spark_session.tar.gz",
            ]:
                source_path = str(self.assets_path / file)
                mlrun.get_dataitem(self.remote_code_dir + file).upload(source_path)
        self.uploaded_code = True

    def _new_function(self, kind, name="run", command=""):
        return mlrun.new_function(
            f"{kind}-{name}",
            kind=kind,
            image=base_image if kind != "local" else None,
            command=command,
        )

    @pytest.mark.parametrize("artifact_format", ["git", "tar.gz", "zip"])
    @pytest.mark.parametrize("codepath", codepaths)
    def test_local_archive(self, artifact_format, codepath):
        workdir, module = codepath
        source = (
            f"{git_uri}#main"
            if artifact_format == "git"
            else str(self.assets_path / f"source_archive.{artifact_format}")
        )
        fn = self._new_function("local")
        fn.with_source_archive(
            source,
            workdir=workdir,
            handler=f"{module}.job_handler",
            target_dir=tempfile.mkdtemp(),
        )
        run = mlrun.run_function(fn)
        assert run.state() == "completed"
        assert run.output("tag")

    @pytest.mark.parametrize("load_mode", ["run", "build"])
    @pytest.mark.parametrize("case", job_cases.keys())
    def test_job_git(self, load_mode, case):
        command, workdir, handler, tag = job_cases[case]
        fn = self._new_function("job", f"{load_mode}-{case}", command)
        fn.with_source_archive(
            f"{git_uri}#{tag}",
            workdir=workdir,
            handler=handler,
            pull_at_runtime=load_mode == "run",
        )
        fn.spec.image_pull_policy = "Always"
        if load_mode == "build":
            mlrun.build_function(fn)
        run = mlrun.run_function(fn)
        assert run.state() == "completed"
        assert run.output("tag") == tag

    @pytest.mark.parametrize("codepath", [(None, "rootfn"), ("subdir", "func")])
    @pytest.mark.parametrize("tag", tags)
    def test_nuclio_deploy(self, codepath, tag):
        workdir, module = codepath
        fn = self._new_function("nuclio")
        fn.with_source_archive(
            f"{git_uri}#{tag}", workdir=workdir, handler=f"{module}:nuclio_handler"
        )
        mlrun.deploy_function(fn)
        resp = fn.invoke("")
        assert resp.decode() == f"tag={tag}"

    def test_serving_deploy(self):
        tag = "main"
        fn = self._new_function("serving")
        fn.with_source_archive(f"{git_uri}#{tag}", handler="srv")
        graph = fn.set_topology("flow")
        graph.to(name="echo", handler="echo").respond()
        mlrun.deploy_function(fn)
        resp = fn.invoke("")
        assert resp.decode() == f"tag={tag}"

    @need_private_git
    def test_private_repo_local(self):
        fn = self._new_function("local", "priv")
        fn.with_source_archive(
            private_repo,
            handler="rootfn.job_handler",
            target_dir=tempfile.mkdtemp(),
        )
        task = mlrun.new_task().with_secrets(
            "inline",
            {
                "GIT_TOKEN": os.environ.get("PRIVATE_GIT_TOKEN", ""),
            },
        )
        run = mlrun.run_function(fn, base_task=task)
        assert run.state() == "completed"
        assert run.output("tag")

    @need_private_git
    @pytest.mark.parametrize("load_mode", ["run", "build"])
    def test_private_repo_job(self, load_mode):
        fn = self._new_function("job", f"{load_mode}-priv")
        fn.with_source_archive(
            private_repo,
            handler="rootfn.job_handler",
            pull_at_runtime=load_mode == "run",
        )
        # fn.spec.image_pull_policy = "Always"
        if load_mode == "build":
            mlrun.build_function(fn)
        run = mlrun.run_function(fn)
        assert run.state() == "completed"
        assert run.output("tag")

    @need_private_git
    def test_private_repo_nuclio(self):
        fn = self._new_function("nuclio", "priv")
        fn.with_source_archive(
            private_repo,
            handler="rootfn:nuclio_handler",
        )
        mlrun.deploy_function(fn)
        resp = fn.invoke("")
        assert "tag=" in resp.decode()

    @pytest.mark.enterprise
    @pytest.mark.parametrize("load_mode", ["run", "build"])
    @pytest.mark.parametrize("compression_format", ["zip", "tar.gz"])
    def test_job_compressed(self, load_mode, compression_format):
        self._upload_code_to_cluster()
        fn = self._new_function("job", f"{load_mode}-compressed")
        fn.with_source_archive(
            self.remote_code_dir + f"source_archive.{compression_format}",
            handler="rootfn.job_handler",
            pull_at_runtime=load_mode == "run",
        )
        if load_mode == "build":
            mlrun.build_function(fn)
        run = mlrun.run_function(fn)
        assert run.state() == "completed"
        assert run.output("tag")

    @pytest.mark.enterprise
    def test_nuclio_tar(self):
        self._upload_code_to_cluster()
        fn = self._new_function("nuclio", "tar")
        fn.with_source_archive(
            self.remote_code_dir + "source_archive.tar.gz",
            handler="rootfn:nuclio_handler",
        )
        fn.verbose = True
        mlrun.deploy_function(fn)
        resp = fn.invoke("")
        assert "tag=" in resp.decode()

    def test_job_project(self):
        project = mlrun.new_project("git-proj-job1", user_project=True)

        # using project.name because this is a user project meaning the project name get concatenated with the user name
        self.custom_project_names_to_delete.append(project.name)
        project.save()
        project.set_source(f"{git_uri}#main", True)  # , workdir="gtst")
        project.set_function(
            name="myjob",
            handler="rootfn.job_handler",
            image=base_image,
            kind="job",
            with_repo=True,
        )

        run = project.run_function("myjob")
        assert run.state() == "completed"
        assert run.output("tag")

    def test_run_function_with_auto_build_and_source_is_idempotent(self):
        project = mlrun.get_or_create_project("run-with-source-and-auto-build")
        # using project.name because this is a user project meaning the project name get concatenated with the user name
        self.custom_project_names_to_delete.append(project.name)
        project.set_source(f"{git_uri}#main", False)
        project.set_function(
            name="myjob",
            handler="rootfn.job_handler",
            image=base_image,
            kind="job",
            with_repo=True,
            requirements=["pandas"],
        )
        project.save()
        project.run_function("myjob", auto_build=True)

        project = mlrun.get_or_create_project("run-with-source-and-auto-build")
        project.set_source(f"{git_uri}#main", False)
        project.set_function(
            name="myjob",
            handler="rootfn.job_handler",
            image=base_image,
            kind="job",
            with_repo=True,
            requirements=["pandas"],
        )
        project.save()
        project.run_function("myjob", auto_build=True)

    def test_run_function_with_auto_build_and_source_is_idempotent_after_failure(self):
        project = mlrun.get_or_create_project("run-with-source-and-auto-build")
        # using project.name because this is a user project meaning the project name get concatenated with the user name
        self.custom_project_names_to_delete.append(project.name)
        project.set_source(f"{git_uri}#main", False)
        project.set_function(
            name="myjob",
            handler="rootfn.job_handler",
            image=base_image,
            kind="job",
            with_repo=True,
            # not existing package, expected to fail when runnning
            requirements=["pandaasdasds"],
        )
        project.save()
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            project.run_function("myjob", auto_build=True)

        project = mlrun.get_or_create_project("run-with-source-and-auto-build")
        project.set_source(f"{git_uri}#main", False)
        project.set_function(
            name="myjob",
            handler="rootfn.job_handler",
            image=base_image,
            kind="job",
            with_repo=True,
            requirements=["pandas"],
        )
        project.save()
        project.run_function("myjob", auto_build=True)

    def test_nuclio_project(self):
        project = mlrun.new_project("git-proj-nuc", user_project=True)
        # using project.name because this is a user project meaning the project name get concatenated with the user name
        self.custom_project_names_to_delete.append(project.name)

        project.save()
        project.set_source(f"{git_uri}#main")
        project.set_function(
            name="mynuclio",
            handler="rootfn:nuclio_handler",
            image=base_image,
            kind="nuclio",
            with_repo=True,
        )

        deployment = project.deploy_function("mynuclio")
        resp = deployment.function.invoke("")
        assert "tag=" in resp.decode()

    def test_project_subdir(self):
        # load project into a tmp dir, look for the project.yaml in the subpath
        project = mlrun.load_project(
            tempfile.mkdtemp(),
            f"{git_uri}#main",
            name="git-proj2",
            user_project=True,
            subpath="subdir",
        )
        # using project.name because this is a user project meaning the project name get concatenated with the user name
        self.custom_project_names_to_delete.append(project.name)

        project.save()
        # run job locally (from cloned source)
        run = project.run_function("myjob", local=True)
        assert run.state() == "completed"
        assert run.output("tag")

        # build and run job on the cluster
        project.build_function("myjob")
        run = project.run_function("myjob")
        assert run.state() == "completed"
        assert run.output("tag")

        # deploy Nuclio function and invoke
        deployment = project.deploy_function("mynuclio")
        resp = deployment.function.invoke("")
        assert "tag=" in resp.decode()

    @pytest.mark.enterprise
    @pytest.mark.parametrize("pull_at_runtime", [False])
    def test_with_igz_spark_from_source(self, pull_at_runtime):
        self._upload_code_to_cluster()
        fn = mlrun.new_function(
            name="spark-test", kind="spark", command="spark_session.py"
        )
        fn.with_igz_spark()

        # spark requires requests
        fn.with_driver_limits(cpu="1300m")
        fn.with_driver_requests(cpu=1, mem="512m")

        fn.with_executor_limits(cpu="1400m")
        fn.with_executor_requests(cpu=1, mem="512m")

        fn.with_source_archive(
            source=self.remote_code_dir + "spark_session.tar.gz",
            pull_at_runtime=pull_at_runtime,
        )
        fn.set_image_pull_configuration(image_pull_policy="Always")

        spark_run = fn.run(auto_build=True)
        assert spark_run.status.state == RunStates.completed
