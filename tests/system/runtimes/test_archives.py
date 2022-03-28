import os
import tempfile

import pytest

import mlrun
import tests.system.base

git_uri = "git://github.com/mlrun/test-git-load.git"
base_image = "mlrun/mlrun"
tags = ["main", "refs/heads/tst"]
codepaths = [(None, "rootfn"), ("subdir", "func")]

cases = {
    # name: (command, workdir, handler, tag)
    "root-hndlr": ("", None, "rootfn.job_handler", tags[0]),
    "subdir-hndlr": ("", "subdir", "func.job_handler", tags[0]),
    "subdir-hndlr-ref": ("", "subdir", "func.job_handler", tags[1]),
    "root-cmd": ("rootfn.py", None, "job_handler", tags[1]),
}


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestGitSource(tests.system.base.TestMLRunSystem):

    project_name = "git-tests"

    @pytest.mark.parametrize("codepath", codepaths)
    def test_local_git(self, codepath):
        workdir, module = codepath
        fn = mlrun.new_function(
            "lcl",
            kind="local",
        )
        fn.with_source_archive(
            f"{git_uri}#main",
            workdir=workdir,
            handler=f"{module}.job_handler",
            target_dir=tempfile.mkdtemp(),
        )
        run = fn.run()
        assert run.state() == "completed"
        assert run.output("tag") == "main"

    @pytest.mark.parametrize("load_mode", ["run", "build"])
    @pytest.mark.parametrize("case", cases.keys())
    def test_job_git(self, load_mode, case):
        os.environ["MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES"] = "false"
        command, workdir, handler, tag = cases[case]
        fn = mlrun.new_function(
            f"job-{load_mode}",
            kind="job",
            image=base_image,
            command=command,
        )
        fn.with_source_archive(
            f"{git_uri}#{tag}",
            workdir=workdir,
            handler=handler,
            pull_at_runtime=load_mode == "run",
        )
        fn.spec.image_pull_policy = "Always"
        if load_mode == "build":
            fn.deploy()
        run = fn.run()
        assert run.state() == "completed"
        assert run.output("tag") == tag

    @pytest.mark.parametrize("codepath", [(None, "rootfn"), ("subdir", "func")])
    @pytest.mark.parametrize("tag", tags)
    def test_nuclio_deploy(self, codepath, tag):
        workdir, module = codepath
        fn = mlrun.new_function(
            "nuclio",
            kind="nuclio",
            image=base_image,
        )
        fn.with_source_archive(
            f"{git_uri}#{tag}", workdir=workdir, handler=f"{module}:nuclio_handler"
        )
        fn.deploy()
        resp = fn.invoke("")
        assert resp.decode() == f"tag={tag}"

    def test_serving_deploy(self):
        tag = "main"
        fn = mlrun.new_function(
            "serving",
            kind="serving",
            image=base_image,
        )
        fn.with_source_archive(f"{git_uri}#{tag}", handler="srv")
        graph = fn.set_topology("flow")
        graph.to(name="echo", handler="echo").respond()
        fn.deploy()
        resp = fn.invoke("")
        assert resp.decode() == f"tag={tag}"
