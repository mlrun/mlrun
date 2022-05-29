import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestKubejobRuntime(tests.system.base.TestMLRunSystem):

    project_name = "kubejob-system-test"

    def test_deploy_function(self):
        code_path = str(self.assets_path / "kubejob_function.py")

        function = mlrun.code_to_function(
            name="simple-function",
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.build_config(base_image="mlrun/mlrun", commands=["pip install pandas"])

        self._logger.debug("Deploying kubejob function")
        function.deploy()

    def test_function_with_param(self):
        code_path = str(self.assets_path / "function_with_params.py")

        proj = mlrun.get_or_create_project(self.project_name, self.results_path)
        project_param = "some value"
        local_param = "my local param"
        proj.spec.params = {"project_param": project_param}
        proj.save()

        function = mlrun.code_to_function(
            name="function-with-params",
            kind="job",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        run = function.run(params={"param1": local_param})
        assert run.status.results["project_param"] == project_param
        assert run.status.results["param1"] == local_param

    def test_class_handler(self):
        code_path = str(self.assets_path / "kubejob_function.py")
        cases = [
            ({"y": 3}, {"rx": 0, "ry": 3, "ra1": 1}),
            ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
        ]
        function = mlrun.code_to_function(
            "function-with-class",
            filename=code_path,
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        for params, results in cases:
            run = function.run(handler="mycls::mtd", params=params)
            print(run.to_yaml())
            assert run.status.results == results

    def test_run_from_module(self):
        function = mlrun.new_function(
            "function-from-module",
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        run = function.run(handler="json.dumps", params={"obj": {"x": 99}})
        print(run.status.results)
        assert run.output("return") == '{"x": 99}'
