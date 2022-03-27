import os
import tempfile

import mlrun
import tests.system.base

git_uri = "git://github.com/mlrun/test-git-load.git"


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestGitSource(tests.system.base.TestMLRunSystem):

    project_name = "git-tests"

    def test_local_load(self):
        fn = mlrun.new_function(
            "lcl",
            kind="local",
        )
        fn.with_source_archive(
            f"{git_uri}#main", "func.job_handler", target_dir=tempfile.mkdtemp()
        )
        # fn.spec.pythonpath =
        fn.spec.workdir = "subdir"
        print(f"clone target dir: {fn.spec.workdir}")
        run = fn.run()
        assert run.state() == "completed"
        assert run.output("tag") == "main"

    def test_job_build(self):
        os.environ["MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES"] = "false"
        tag = "main"
        fn = mlrun.new_function(
            "job1",
            kind="job",
            image="yhaviv/mlrun:1.0.0-rc20",
        )
        fn.with_source_archive(
            f"{git_uri}#{tag}",
            "func.job_handler",
            workdir="subdir",
            pull_at_runtime=False,
        )
        fn.spec.image_pull_policy = "Always"
        print(fn.to_yaml())
        fn.deploy()
        run = fn.run()
        print(run.to_yaml())
        assert run.state() == "completed"
        assert run.output("tag") == tag

    def test_nuclio_deploy(self):
        tag = "main"
        fn = mlrun.new_function(
            "nuclio1",
            kind="nuclio",
            image="mlrun/mlrun",
        )
        fn.with_source_archive(
            f"{git_uri}#{tag}", "func:nuclio_handler", workdir="subdir"
        )
        print(fn.to_yaml())
        fn.deploy()
        resp = fn.invoke("")
        assert resp.decode() == f"tag={tag}"
