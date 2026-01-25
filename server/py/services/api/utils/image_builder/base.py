# Copyright 2026 Iguazio
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


import abc
import typing

from kubernetes import client

import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.runtimes.pod

import framework.api.utils
import framework.utils.singletons.k8s


class AbstractBaseImageBuilder(abc.ABC):
    @abc.abstractmethod
    def make_build_pod(
        self,
        project: str,
        context: str,
        dest: str,
        dockerfile: typing.Optional[str] = None,
        dockertext: typing.Optional[str] = None,
        inline_code: typing.Optional[str] = None,
        inline_path: typing.Optional[str] = None,
        requirements: typing.Optional[list] = None,
        requirements_path: typing.Optional[str] = None,
        secret_name: typing.Optional[str] = None,
        name: str = "",
        verbose: bool = False,
        builder_env: typing.Optional[list[client.V1EnvVar]] = None,
        runtime_spec=None,
        registry: typing.Optional[str] = None,
        extra_args: str = "",
        extra_labels: typing.Optional[dict] = None,
        project_secrets: typing.Optional[list[client.V1EnvVar]] = None,
        project_default_fucntion_node_selector: typing.Optional[dict] = None,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
    ) -> framework.utils.singletons.k8s.BasePod:
        """Create the Kubernetes pod spec for building the image."""
        raise NotImplementedError()


class BaseImageBuilder(abc.ABC):
    def get_kaniko_spec_attributes_from_runtime(
        self,
        project: str,
        runtime_spec: mlrun.runtimes.pod.KubeResourceSpec,
        project_default_fucntion_node_selector: typing.Optional[dict] = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ) -> dict[str, typing.Callable[[typing.Any], typing.Any]]:
        """
        Get the names of spec attributes that are defined
        for runtime but should also be applied to Builder pod.
        """

        project_default_fucntion_node_selector = (
            project_default_fucntion_node_selector or {}
        )

        # preemption mode scheduling constraints cache
        _preemption_enrichment_result: dict = {}

        def service_account_handler(attr_value):
            (
                allowed_service_accounts,
                forbidden_service_accounts,
                default_service_account,
            ) = framework.api.utils.resolve_project_service_account_details(
                project, auth_info=auth_info
            )
            if attr_value:
                runtime_spec.validate_service_account(
                    allowed_service_accounts, forbidden_service_accounts
                )
            else:
                attr_value = default_service_account
            return attr_value

        def get_merged_node_selector(attr_value):
            attr_value = mlrun.utils.to_non_empty_values_dict(
                mlrun.utils.helpers.merge_dicts_with_precedence(
                    mlrun.mlconf.get_default_function_node_selector(),
                    project_default_fucntion_node_selector,
                    attr_value,
                )
            )
            return attr_value

        def preemption_mode_handler(key):
            if key not in _preemption_enrichment_result:
                keys = ["node_selector", "tolerations", "affinity"]
                values = mlrun.k8s_utils.enrich_preemption_mode(
                    preemption_mode=runtime_spec.preemption_mode,
                    node_selector=get_merged_node_selector(runtime_spec.node_selector),
                    affinity=runtime_spec.affinity,
                    tolerations=runtime_spec.tolerations,
                )
                _preemption_enrichment_result.update(dict(zip(keys, values)))
            return _preemption_enrichment_result[key]

        def node_selector_handler(attr_value):
            return preemption_mode_handler("node_selector")

        def affinity_handler(attr_value):
            return preemption_mode_handler("affinity")

        def tolerations_handler(attr_value):
            return preemption_mode_handler("tolerations")

        def identity_handler(attr_value):
            return attr_value

        return {
            "node_name": identity_handler,
            "node_selector": node_selector_handler,
            "affinity": affinity_handler,
            "tolerations": tolerations_handler,
            "priority_class_name": identity_handler,
            "service_account": service_account_handler,
        }
