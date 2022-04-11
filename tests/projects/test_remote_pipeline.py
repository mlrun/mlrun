import base64
import json
import unittest.mock

import kfp.compiler
import kubernetes
import kubernetes.client

import mlrun
import tests.projects.base_pipeline


class TestRemotePipeline(tests.projects.base_pipeline.TestPipeline):
    def _set_functions(self):
        self.project.set_function(
            func=f"{self.assets_path}/remote_pipeline.py",
            name="func1",
            image="mlrun/mlrun",
            handler="func1",
        )
        self.project.set_function(
            func=f"{self.assets_path}/remote_pipeline.py",
            name="func2",
            image="mlrun/mlrun",
            handler="func2",
        )

    def test_kfp_pipeline_enriched_with_priority_class_name(self, rundb_mock):
        mlrun.config.config.default_function_priority_class_name = "default-high"

        mlrun.projects.pipeline_context.clear(with_project=True)
        self.project = self._create_project("remotepipe")
        self._set_functions()
        self.project.set_workflow(
            "p1",
            workflow_path=str(f'{self.assets_path / "remote_pipeline.py"}'),
            handler="kfp_pipeline",
            engine="kfp",
            local=False,
        )
        self.project.save()

        # we are monkey patching kfp.compiler.Compiler._create_workflow,
        # in the monkey patching we are enriching the pipeline step with the functions priority_class_name
        # after enriching the compile function passes the enriched workflow to
        # kfp.compiler.Compiler._write_workflow and that's why we mock _write_workflow to get the
        # passed args which one of them is workflow
        kfp.compiler.Compiler._write_workflow = unittest.mock.Mock(return_value=True)
        self.project.save_workflow(
            "p1",
            target=self.project.artifact_path,
        )

        workflow = kfp.compiler.Compiler._write_workflow.call_args_list[0][0][0]
        for step in workflow["spec"]["templates"]:
            if step.get("container"):
                assert (
                    step["PriorityClassName"]
                    == mlrun.config.config.default_function_priority_class_name
                )

    def test_kfp_pipeline_enriched_with_affinity_and_tolerations_enriched_by_preemption_mode(
        self, rundb_mock
    ):
        k8s_api = kubernetes.client.ApiClient()
        mlrun.mlconf.default_function_priority_class_name = "default-high"

        node_selector = {"label-1": "val1"}
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )

        preemptible_tolerations = [
            kubernetes.client.V1Toleration(
                effect="NoSchedule",
                key="test1",
                operator="Exists",
            )
        ]
        serialized_tolerations = k8s_api.sanitize_for_serialization(
            preemptible_tolerations
        )
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = "constrain"
        mlrun.projects.pipeline_context.clear(with_project=True)
        self.project = self._create_project("remotepipe")
        self._set_functions()
        self.project.set_workflow(
            "p1",
            workflow_path=str(f'{self.assets_path / "remote_pipeline.py"}'),
            handler="kfp_pipeline",
            engine="kfp",
            local=False,
        )
        self.project.save()

        kfp.compiler.Compiler._write_workflow = unittest.mock.Mock(return_value=True)
        self.project.save_workflow(
            "p1",
            target=self.project.artifact_path,
        )

        workflow = kfp.compiler.Compiler._write_workflow.call_args_list[0][0][0]
        expected_tolerations = [
            {"effect": "NoSchedule", "key": "test1", "operator": "Exists"}
        ]
        expected_preemptible_affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "label-1",
                                    "operator": "In",
                                    "values": ["val1"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
        for step in workflow["spec"]["templates"]:
            if step.get("container"):
                print(step)
                assert (
                    step["PriorityClassName"]
                    == mlrun.config.config.default_function_priority_class_name
                )
                assert step["affinity"] == expected_preemptible_affinity
                assert step["tolerations"] == expected_tolerations
