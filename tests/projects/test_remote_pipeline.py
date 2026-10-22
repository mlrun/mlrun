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

import base64
import copy
import json
import os
import pathlib
import sys

import kubernetes
import kubernetes.client
import pytest
import yaml

import mlrun
import mlrun.config
import tests.projects.assets.remote_pipeline_with_overridden_resources
import tests.projects.base_pipeline
from mlrun.common.schemas import SecurityContextEnrichmentModes


@pytest.fixture()
def workflow_path():
    workflow_path = (
        pathlib.Path(sys.modules[TestRemotePipeline.__module__].__file__)
        .absolute()
        .parent
        / "workpipe.yaml"
    )
    yield workflow_path

    # remove generated workflow file
    if os.path.exists(workflow_path):
        os.remove(workflow_path)


class TestRemotePipeline(tests.projects.base_pipeline.TestPipeline):
    pipeline_handler = "kfp_pipeline"

    def _get_functions(self):
        func1 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func1",
            image="mlrun/mlrun",
            kind="job",
        )
        func2 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func2",
            image="mlrun/mlrun",
            kind="job",
        )

        func3 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func3",
            image="mlrun/mlrun",
            kind="job",
        )
        func4 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func4",
            image="mlrun/mlrun",
            kind="nuclio",
        )
        return func1, func2, func3, func4

    def test_kfp_pipeline_enriched_with_priority_class_name(
        self, rundb_mock, workflow_path
    ):
        self.pipeline_path = "remote_pipeline.py"
        mlrun.projects.pipeline_context.clear(with_project=True)

        mlrun.mlconf.valid_function_priority_class_names = (
            "default-high,default-medium,default-low"
        )
        self._create_project("remotepipe")
        func1, func2, func3, func4 = self._get_functions()

        func1.with_priority_class("default-high")
        func2.with_priority_class("default-low")

        self.project.set_function(func1)
        self.project.set_function(func2)
        self.project.set_function(func3)
        self.project.set_function(func4)

        self.project.set_workflow(
            "p1",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            handler=self.pipeline_handler,
            engine="kfp",
            local=False,
        )
        self.project.save()

        # we are monkey patching kfp.compiler.Compiler._create_workflow,
        # in the monkey patching we are enriching the pipeline step with the functions priority_class_name
        # after enriching the compile function passes the enriched workflow to
        # kfp.compiler.Compiler._write_workflow and that's why we mock _write_workflow to get the
        # passed args which one of them is workflow
        # kfp.compiler.Compiler._write_workflow = unittest.mock.Mock(return_value=True)
        self.project.save_workflow(
            "p1",
            target=str(workflow_path),
        )
        with workflow_path.open() as workflow_file:
            workflow = yaml.safe_load(workflow_file)
            for step in workflow["spec"]["templates"]:
                if step.get("container") and step.get("name"):
                    # the step name is constructed from the function name and the handler
                    if step.get("name") == "func1-func1":
                        assert step.get("PriorityClassName") == "default-high"
                    elif step.get("name") == "func2-func2":
                        assert step.get("PriorityClassName") == "default-low"

    # Will be fixed in a follow up PR handling KFP pods
    # def test_kfp_pipeline_enriched_with_affinity_and_tolerations_enriched_by_preemption_mode(
    #     self, rundb_mock, workflow_path
    # ):
    #     self.pipeline_path = "remote_pipeline.py"
    #     mlrun.projects.pipeline_context.clear(with_project=True)
    #     k8s_api = kubernetes.client.ApiClient()
    #
    #     node_selector = {"label-1": "val1"}
    #     mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
    #         json.dumps(node_selector).encode("utf-8")
    #     )
    #
    #     preemptible_tolerations = [
    #         kubernetes.client.V1Toleration(
    #             effect="NoSchedule",
    #             key="test1",
    #             operator="Exists",
    #         )
    #     ]
    #     serialized_tolerations = k8s_api.sanitize_for_serialization(
    #         preemptible_tolerations
    #     )
    #     mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
    #         json.dumps(serialized_tolerations).encode("utf-8")
    #     )
    #
    #     self._create_project("remotepipe")
    #     func1, func2, func3, func4 = self._get_functions()
    #
    #     func1.with_preemption_mode("constrain")
    #     func2.with_preemption_mode("prevent")
    #     func3.with_preemption_mode("constrain")
    #     func4.with_preemption_mode("allow")
    #
    #     self.project.set_function(func1)
    #     self.project.set_function(func2)
    #     self.project.set_function(func3)
    #     self.project.set_function(func4)
    #
    #     self.project.set_workflow(
    #         "p1",
    #         workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
    #         handler=self.pipeline_handler,
    #         engine="kfp",
    #         local=False,
    #     )
    #     self.project.save()
    #
    #     self.project.save_workflow(
    #         "p1",
    #         target=str(workflow_path),
    #     )
    #
    #     with workflow_path.open() as workflow_file:
    #         workflow = yaml.safe_load(workflow_file)
    #         for step in workflow["spec"]["templates"]:
    #             if step.get("container") and step.get("name"):
    #                 # the step name is constructed from the function name and the handler
    #                 if step.get("name") == "func1-func1":
    #                     # expects constrain
    #                     assert step.get("affinity") == self._get_preemptible_affinity()
    #                     assert (
    #                         step.get("tolerations")
    #                         == self._get_preemptible_tolerations()
    #                     )
    #                 elif step.get("name") == "func2-func1":
    #                     # expects prevent
    #                     assert step.get("affinity") is None
    #                     assert step.get("tolerations") is None
    #                 elif step.get("name") == "deploy-func3":
    #                     # expects constrain
    #                     assert step.get("affinity") == self._get_preemptible_affinity()
    #                     assert (
    #                         step.get("tolerations")
    #                         == self._get_preemptible_tolerations()
    #                     )
    #                 elif step.get("name") == "deploy-func4":
    #                     # expects allow
    #                     assert (
    #                         step.get("tolerations")
    #                         == self._get_preemptible_tolerations()
    #                     )
    #                 else:
    #                     raise mlrun.errors.MLRunRuntimeError(
    #                         "You missed a container to test"
    #                     )

    @pytest.mark.parametrize(
        "enrichment_mode,kfp_pod_user_unix_id,enrichment_group_id_override,expected_security_context,expect_error",
        [
            # enrichment mode is disabled, security context should not be added
            (SecurityContextEnrichmentModes.disabled.value, None, None, None, False),
            # enrichment mode is override and kfp pod user id is not set, should raise error
            (SecurityContextEnrichmentModes.override.value, None, None, None, True),
            # user id 0 (root) is not allowed
            (SecurityContextEnrichmentModes.override.value, 0, None, None, True),
            # security context should be enriched
            (
                SecurityContextEnrichmentModes.override.value,
                5,
                None,
                {
                    "runAsUser": 5,
                    "runAsGroup": mlrun.mlconf.function.spec.security_context.enrichment_group_id,
                },
                False,
            ),
            # group id should be enriched with kfp pod user id
            (
                SecurityContextEnrichmentModes.override.value,
                5,
                -1,
                {"runAsUser": 5, "runAsGroup": 5},
                False,
            ),
        ],
    )
    def test_kfp_pipeline_enriched_with_security_context(
        self,
        rundb_mock,
        workflow_path,
        enrichment_mode,
        kfp_pod_user_unix_id,
        enrichment_group_id_override,
        expected_security_context,
        expect_error,
    ):
        self.pipeline_path = "remote_pipeline.py"
        mlrun.projects.pipeline_context.clear(with_project=True)

        self._create_project("remotepipe")
        func1, func2, func3, func4 = self._get_functions()

        self.project.set_function(func1)
        self.project.set_function(func2)
        self.project.set_function(func3)
        self.project.set_function(func4)

        self.project.set_workflow(
            "p1",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            handler=self.pipeline_handler,
            engine="kfp",
            local=False,
        )

        mlrun.mlconf.function.spec.security_context.enrichment_mode = enrichment_mode
        mlrun.mlconf.function.spec.security_context.pipelines.kfp_pod_user_unix_id = (
            kfp_pod_user_unix_id
        )
        if enrichment_group_id_override:
            mlrun.mlconf.function.spec.security_context.enrichment_group_id = (
                enrichment_group_id_override
            )
        self.project.save()

        if expect_error:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
                self.project.save_workflow(
                    "p1",
                    target=str(workflow_path),
                )
            assert (
                f"Kubeflow pipeline pod user id is invalid: {kfp_pod_user_unix_id}, "
                f"it must be an integer greater than 0. "
                "See mlrun.config.function.spec.security_context.pipelines.kfp_pod_user_unix_id for more details."
                in str(exc.value)
            )

        else:
            self.project.save_workflow(
                "p1",
                target=str(workflow_path),
            )
            with workflow_path.open() as workflow_file:
                workflow = yaml.safe_load(workflow_file)
                for step in workflow["spec"]["templates"]:
                    if step.get("container") and step.get("name"):
                        assert (
                            step.get("SecurityContext") == expected_security_context
                        ), (
                            f"security context was not enriched correctly in step: {step.get('name')}"
                        )

    @pytest.mark.parametrize(
        "enrichment_mode,function_kinds,expected_homes",
        [
            # several non-local functions - each enriched step gets exactly one default HOME,
            # and an explicit HOME on the step template wins
            (
                SecurityContextEnrichmentModes.override.value,
                ["job", "job", "job", "nuclio"],
                {"func1": "/home/custom", "func2": "/tmp"},
            ),
            # no security context is enriched - only the explicit HOME remains
            (
                SecurityContextEnrichmentModes.disabled.value,
                ["job", "job", "job", "nuclio"],
                {"func1": "/home/custom", "func2": None},
            ),
        ],
    )
    def test_kfp_pipeline_enriched_with_non_root_home(
        self,
        rundb_mock,
        workflow_path,
        enrichment_mode,
        function_kinds,
        expected_homes,
    ):
        mlrun.mlconf.function.spec.security_context.enrichment_mode = enrichment_mode

        homes = self._compile_workflow_step_homes(workflow_path, function_kinds)

        assert homes == expected_homes

    @pytest.mark.parametrize(
        "synced_security_context, expected_home",
        [
            ({"non_root_home": "/home/custom"}, "/home/custom"),
            ({"non_root_home": ""}, None),
            # older servers don't send the key - fall back to the SDK default
            ({}, "/tmp"),
        ],
    )
    def test_kfp_pipeline_non_root_home_after_client_spec_sync(
        self, rundb_mock, workflow_path, synced_security_context, expected_home
    ):
        # the security context as synced by HTTPRunDB.connect() from the server's client spec
        mlrun.mlconf.function.spec.security_context = {
            "default": "e30=",
            "enrichment_mode": SecurityContextEnrichmentModes.override.value,
            "enrichment_group_id": 65534,
            "pipelines": {"kfp_pod_user_unix_id": 5},
            **synced_security_context,
        }

        try:
            homes = self._compile_workflow_step_homes(workflow_path, ["job", "job"])

            assert homes == {"func1": "/home/custom", "func2": expected_home}
        finally:
            # the config reload doesn't restore keys missing from the synced security context
            mlrun.mlconf.function.spec.security_context = copy.deepcopy(
                mlrun.config.default_config["function"]["spec"]["security_context"]
            )

    def test_kfp_pod_of_local_function_not_enriched_with_non_root_home(self):
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            SecurityContextEnrichmentModes.override.value
        )
        function = mlrun.new_function("local-func", kind="local")
        kfp_pod_template = {"name": "step", "container": {"env": []}}

        mlrun.projects.pipelines._enrich_kfp_pod_security_context(
            kfp_pod_template, function
        )

        assert kfp_pod_template == {"name": "step", "container": {"env": []}}

    def test_kfp_pod_non_root_home_enrichment_is_idempotent(self):
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            SecurityContextEnrichmentModes.override.value
        )
        kfp_pod_template = {"name": "step", "container": {}}

        # the patcher applies every project function to every step template
        for name in ["func1", "func2"]:
            mlrun.projects.pipelines._enrich_kfp_pod_security_context(
                kfp_pod_template, mlrun.new_function(name, kind="job")
            )

        assert kfp_pod_template["container"]["env"] == [
            {"name": "HOME", "value": "/tmp"}
        ]

    def _compile_workflow_step_homes(self, workflow_path, function_kinds):
        self.pipeline_path = "remote_pipeline_with_home.py"
        mlrun.projects.pipeline_context.clear(with_project=True)

        self._create_project("remotepipe")
        for index, kind in enumerate(function_kinds):
            self.project.set_function(
                mlrun.new_function(
                    source=f"{self.assets_path}/{self.pipeline_path}",
                    name=f"func{index + 1}",
                    image="mlrun/mlrun",
                    kind=kind,
                )
            )
        self.project.set_workflow(
            "p1",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            handler=self.pipeline_handler,
            engine="kfp",
            local=False,
        )
        self.project.save()

        self.project.save_workflow("p1", target=str(workflow_path))

        with workflow_path.open() as workflow_file:
            workflow = yaml.safe_load(workflow_file)
        homes = {}
        for step in workflow["spec"]["templates"]:
            if not step.get("container"):
                continue
            home_values = [
                env_var["value"]
                for env_var in step["container"].get("env") or []
                if env_var["name"] == "HOME"
            ]
            assert len(home_values) <= 1, (
                f"step {step['name']} has multiple HOME env vars: {home_values}"
            )
            homes[step["name"].split("-")[0]] = home_values[0] if home_values else None
        return homes

    def _get_preemptible_tolerations(self):
        return [{"effect": "NoSchedule", "key": "test1", "operator": "Exists"}]

    def _get_preemptible_affinity(self):
        return {
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

    def test_kfp_pipeline_overwrites_enriched_attributes(
        self, rundb_mock, workflow_path
    ):
        mlrun.projects.pipeline_context.clear(with_project=True)
        k8s_api = kubernetes.client.ApiClient()
        self.pipeline_path = "remote_pipeline_with_overridden_resources.py"
        default_function_pod_resources = {
            "requests": {"cpu": "25m", "memory": "1Mi"},
            "limits": {"cpu": "2", "memory": "20Gi"},
        }
        mlrun.mlconf.default_function_pod_resources = default_function_pod_resources

        node_selector = {"label-1": "val1"}
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )

        self._create_project("remotepipe")
        func1 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func1",
            image="mlrun/mlrun",
            kind="job",
        )
        func2 = mlrun.new_function(
            source=f"{self.assets_path}/{self.pipeline_path}",
            name="func2",
            image="mlrun/mlrun",
            kind="job",
        )

        func2.with_preemption_mode("constrain")
        self.project.set_function(func1)
        self.project.set_function(func2)

        self.project.set_workflow(
            "p1",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            handler=self.pipeline_handler,
            engine="kfp",
            local=False,
        )
        self.project.save()

        self.project.save_workflow(
            "p1",
            target=str(workflow_path),
        )

        with workflow_path.open() as workflow_file:
            workflow = yaml.safe_load(workflow_file)
            for step in workflow["spec"]["templates"]:
                if step.get("container") and step.get("name"):
                    # the step name is constructed from the function name and the handler
                    if step.get("name") == "func1-func1":
                        assert (
                            step["container"]["resources"]["requests"]
                            == mlrun.mlconf.default_function_pod_resources.requests.to_dict()
                        )
                        assert step["container"]["resources"]["limits"] == {
                            "cpu": "2000m",
                            "memory": "4G",
                        }
                    elif step.get("name") == "func2-func1":
                        assert step["affinity"] == k8s_api.sanitize_for_serialization(
                            tests.projects.assets.remote_pipeline_with_overridden_resources.overridden_affinity
                        )


@pytest.mark.parametrize("workflow_name", [("test-test"), ("new-1-main"), ("test")])
def test_workflow_name(workflow_name):
    project_name = "test"
    # When creating a schedule workflow we adding the project name to the workflow name.
    before_renaming = f"{project_name}-{workflow_name}"
    assert workflow_name == mlrun.utils.normalize_workflow_name(
        before_renaming, project_name
    )
