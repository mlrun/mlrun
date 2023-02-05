# Copyright 2018 Iguazio
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
import inspect
import os
import typing
import uuid
from enum import Enum

import dotenv
import kfp.dsl
import kubernetes.client as k8s_client
from deprecated import deprecated

import mlrun.errors
import mlrun.utils.regex

from ..api.schemas import (
    NodeSelectorOperator,
    PreemptionModes,
    SecurityContextEnrichmentModes,
)
from ..config import config as mlconf
from ..k8s_utils import (
    generate_preemptible_node_selector_requirements,
    generate_preemptible_nodes_affinity_terms,
    generate_preemptible_nodes_anti_affinity_terms,
    generate_preemptible_tolerations,
)
from ..secrets import SecretsStore
from ..utils import logger, normalize_name, update_in
from .base import BaseRuntime, FunctionSpec, spec_fields
from .utils import (
    apply_kfp,
    get_gpu_from_resource_requirement,
    get_item_name,
    get_resource_labels,
    set_named_item,
    verify_limits,
    verify_requests,
)

sanitized_types = {
    "affinity": {
        "attribute_type_name": "V1Affinity",
        "attribute_type": k8s_client.V1Affinity,
        "sub_attribute_type": None,
        "contains_many": False,
        "not_sanitized_class": dict,
    },
    "tolerations": {
        "attribute_type_name": "List[V1.Toleration]",
        "attribute_type": list,
        "contains_many": True,
        "sub_attribute_type": k8s_client.V1Toleration,
        "not_sanitized_class": list,
    },
    "security_context": {
        "attribute_type_name": "V1SecurityContext",
        "attribute_type": k8s_client.V1SecurityContext,
        "sub_attribute_type": None,
        "contains_many": False,
        "not_sanitized_class": dict,
    },
}

sanitized_attributes = {
    "affinity": sanitized_types["affinity"],
    "tolerations": sanitized_types["tolerations"],
    "security_context": sanitized_types["security_context"],
    "executor_tolerations": sanitized_types["tolerations"],
    "driver_tolerations": sanitized_types["tolerations"],
    "executor_affinity": sanitized_types["affinity"],
    "driver_affinity": sanitized_types["affinity"],
    "executor_security_context": sanitized_types["security_context"],
    "driver_security_context": sanitized_types["security_context"],
}


class KubeResourceSpec(FunctionSpec):
    _dict_fields = spec_fields + [
        "volumes",
        "volume_mounts",
        "env",
        "resources",
        "replicas",
        "image_pull_policy",
        "service_account",
        "image_pull_secret",
        "node_name",
        "node_selector",
        "affinity",
        "priority_class_name",
        "tolerations",
        "preemption_mode",
        "security_context",
    ]

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        default_handler=None,
        pythonpath=None,
        entry_points=None,
        description=None,
        workdir=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        build=None,
        image_pull_secret=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        disable_auto_mount=False,
        priority_class_name=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
            pythonpath=pythonpath,
            disable_auto_mount=disable_auto_mount,
        )
        self._volumes = {}
        self._volume_mounts = {}
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self._resources = self.enrich_resources_with_default_pod_resources(
            "resources", resources
        )

        self.replicas = replicas
        self.image_pull_policy = image_pull_policy
        # default service account is set in mlrun.utils.process_function_service_account
        # due to project specific defaults
        self.service_account = service_account
        self.image_pull_secret = (
            image_pull_secret or mlrun.mlconf.function.spec.image_pull_secret.default
        )
        self.node_name = node_name
        self.node_selector = (
            node_selector or mlrun.mlconf.get_default_function_node_selector()
        )
        self._affinity = affinity
        self.priority_class_name = (
            priority_class_name or mlrun.mlconf.default_function_priority_class_name
        )
        self._tolerations = tolerations
        self.preemption_mode = preemption_mode
        self.security_context = (
            security_context or mlrun.mlconf.get_default_function_security_context()
        )

    @property
    def volumes(self) -> list:
        return list(self._volumes.values())

    @volumes.setter
    def volumes(self, volumes):
        self._volumes = {}
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

    @property
    def volume_mounts(self) -> list:
        return list(self._volume_mounts.values())

    @volume_mounts.setter
    def volume_mounts(self, volume_mounts):
        self._volume_mounts = {}
        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(volume_mount)

    @property
    def affinity(self) -> k8s_client.V1Affinity:
        return self._affinity

    @affinity.setter
    def affinity(self, affinity):
        self._affinity = transform_attribute_to_k8s_class_instance("affinity", affinity)

    @property
    def tolerations(self) -> typing.List[k8s_client.V1Toleration]:
        return self._tolerations

    @tolerations.setter
    def tolerations(self, tolerations):
        self._tolerations = transform_attribute_to_k8s_class_instance(
            "tolerations", tolerations
        )

    @property
    def resources(self) -> dict:
        return self._resources

    @resources.setter
    def resources(self, resources):
        self._resources = self.enrich_resources_with_default_pod_resources(
            "resources", resources
        )

    @property
    def preemption_mode(self) -> str:
        return self._preemption_mode

    @preemption_mode.setter
    def preemption_mode(self, mode):
        self._preemption_mode = mode or mlconf.function_defaults.preemption_mode
        self.enrich_function_preemption_spec()

    @property
    def security_context(self) -> k8s_client.V1SecurityContext:
        return self._security_context

    @security_context.setter
    def security_context(self, security_context):
        self._security_context = transform_attribute_to_k8s_class_instance(
            "security_context", security_context
        )

    def to_dict(self, fields=None, exclude=None):
        exclude = exclude or []
        _exclude = ["affinity", "tolerations", "security_context"]
        struct = super().to_dict(fields, exclude=list(set(exclude + _exclude)))
        api = k8s_client.ApiClient()
        for field in _exclude:
            if field not in exclude:
                struct[field] = api.sanitize_for_serialization(getattr(self, field))
        return struct

    def update_vols_and_mounts(
        self, volumes, volume_mounts, volume_mounts_field_name="_volume_mounts"
    ):
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(volume_mount, volume_mounts_field_name)

    def validate_service_account(self, allowed_service_accounts):
        if (
            allowed_service_accounts
            and self.service_account not in allowed_service_accounts
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Function service account {self.service_account} is not in allowed "
                + f"service accounts {allowed_service_accounts}"
            )

    def _get_affinity_as_k8s_class_instance(self):
        pass

    def _set_volume_mount(
        self, volume_mount, volume_mounts_field_name="_volume_mounts"
    ):
        # using the mountPath as the key cause it must be unique (k8s limitation)
        # volume_mount may be an V1VolumeMount instance (object access, snake case) or sanitized dict (dict
        # access, camel case)
        getattr(self, volume_mounts_field_name)[
            get_item_name(volume_mount, "mountPath")
            or get_item_name(volume_mount, "mount_path")
        ] = volume_mount

    def _verify_and_set_limits(
        self,
        resources_field_name,
        mem: str = None,
        cpu: str = None,
        gpus: int = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        resources = verify_limits(
            resources_field_name, mem=mem, cpu=cpu, gpus=gpus, gpu_type=gpu_type
        )
        if not patch:
            update_in(
                getattr(self, resources_field_name),
                "limits",
                resources,
            )
        else:
            for resource, resource_value in resources.items():
                # gpu_type can contain "." (e.g nvidia.com/gpu) in the name which will result the `update_in` to split
                # the resource name
                if resource == gpu_type:
                    limits: dict = getattr(self, resources_field_name).setdefault(
                        "limits", {}
                    )
                    limits.update({resource: resource_value})
                else:
                    update_in(
                        getattr(self, resources_field_name),
                        f"limits.{resource}",
                        resource_value,
                    )

    def _verify_and_set_requests(
        self,
        resources_field_name,
        mem: str = None,
        cpu: str = None,
        patch: bool = False,
    ):
        resources = verify_requests(resources_field_name, mem=mem, cpu=cpu)
        if not patch:
            update_in(
                getattr(self, resources_field_name),
                "requests",
                resources,
            )
        else:
            for resource, resource_value in resources.items():
                update_in(
                    getattr(self, resources_field_name),
                    f"requests.{resource}",
                    resource_value,
                )

    def with_limits(
        self,
        mem: str = None,
        cpu: str = None,
        gpus: int = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set pod cpu/memory/gpu limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        self._verify_and_set_limits("resources", mem, cpu, gpus, gpu_type, patch=patch)

    def with_requests(self, mem: str = None, cpu: str = None, patch: bool = False):
        """
        set requested (desired) pod cpu/memory resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self._verify_and_set_requests("resources", mem, cpu, patch)

    def enrich_resources_with_default_pod_resources(
        self, resources_field_name: str, resources: dict
    ):
        resources_types = ["cpu", "memory"]
        resource_requirements = ["requests", "limits"]
        default_resources = mlconf.get_default_function_pod_resources()

        if resources:
            for resource_requirement in resource_requirements:
                for resource_type in resources_types:
                    if (
                        resources.setdefault(resource_requirement, {}).setdefault(
                            resource_type
                        )
                        is None
                    ):
                        resources[resource_requirement][
                            resource_type
                        ] = default_resources[resource_requirement][resource_type]
        # This enables the user to define that no defaults would be applied on the resources
        elif resources == {}:
            return resources
        else:
            resources = default_resources
        resources["requests"] = verify_requests(
            resources_field_name,
            mem=resources["requests"]["memory"],
            cpu=resources["requests"]["cpu"],
        )
        gpu_type, gpu_value = get_gpu_from_resource_requirement(resources["limits"])
        resources["limits"] = verify_limits(
            resources_field_name,
            mem=resources["limits"]["memory"],
            cpu=resources["limits"]["cpu"],
            gpus=gpu_value,
            gpu_type=gpu_type,
        )
        if not resources["requests"] and not resources["limits"]:
            return {}
        return resources

    def _merge_node_selector(self, node_selector: typing.Dict[str, str]):
        if not node_selector:
            return

        # merge node selectors - precedence to existing node selector
        self.node_selector = {**node_selector, **self.node_selector}

    def _merge_tolerations(
        self,
        tolerations: typing.List[k8s_client.V1Toleration],
        tolerations_field_name: str,
    ):
        if not tolerations:
            return
        # In case function has no toleration, take all from input
        self_tolerations = getattr(self, tolerations_field_name)
        if not self_tolerations:
            setattr(self, tolerations_field_name, tolerations)
            return
        tolerations_to_add = []

        # Only add non-matching tolerations to avoid duplications
        for toleration in tolerations:
            to_add = True
            for function_toleration in self_tolerations:
                if function_toleration == toleration:
                    to_add = False
                    break
            if to_add:
                tolerations_to_add.append(toleration)

        if len(tolerations_to_add) > 0:
            self_tolerations.extend(tolerations_to_add)

    def _override_required_during_scheduling_ignored_during_execution(
        self,
        node_selector: k8s_client.V1NodeSelector,
        affinity_field_name: str,
    ):
        self._initialize_affinity(affinity_field_name)
        self._initialize_node_affinity(affinity_field_name)

        self_affinity = getattr(self, affinity_field_name)
        self_affinity.node_affinity.required_during_scheduling_ignored_during_execution = (
            node_selector
        )

    def enrich_function_preemption_spec(
        self,
        preemption_mode_field_name: str = "preemption_mode",
        tolerations_field_name: str = "tolerations",
        affinity_field_name: str = "affinity",
        node_selector_field_name: str = "node_selector",
    ):
        """
        Enriches function pod with the below described spec.
        If no preemptible node configuration is provided, do nothing.
            `allow` 	- Adds Tolerations if configured.
                          otherwise, assume pods can be scheduled on preemptible nodes.
                        > Purges any `affinity` / `anti-affinity` preemption related configuration
                        > Purges preemptible node selector
            `constrain` - Uses node-affinity to make sure pods are assigned using OR on the configured
                          node label selectors.
                        > Merges tolerations with preemptible tolerations.
                        > Purges any `anti-affinity` preemption related configuration
            `prevent`	- Prevention is done either using taints (if Tolerations were configured) or anti-affinity.
                        > Purges any `tolerations` preemption related configuration
                        > Purges any `affinity` preemption related configuration
                        > Purges preemptible node selector
                        > Sets anti-affinity and overrides any affinity if no tolerations were configured
            `none`      - Doesn't apply any preemptible node selection configuration.
        """
        # nothing to do here, configuration is not populated
        if not mlconf.is_preemption_nodes_configured():
            return

        if not getattr(self, preemption_mode_field_name):
            # We're not supposed to get here, but if we do, we'll set the private attribute to
            # avoid triggering circular enrichment.
            setattr(
                self,
                f"_{preemption_mode_field_name}",
                mlconf.function_defaults.preemption_mode,
            )
            logger.debug(
                "No preemption mode was given, using the default preemption mode",
                default_preemption_mode=getattr(self, preemption_mode_field_name),
            )
        self_preemption_mode = getattr(self, preemption_mode_field_name)
        # don't enrich with preemption configuration.
        if self_preemption_mode == PreemptionModes.none.value:
            return
        # remove preemptible tolerations and remove preemption related configuration
        # and enrich with anti-affinity if preemptible tolerations configuration haven't been provided
        if self_preemption_mode == PreemptionModes.prevent.value:
            # ensure no preemptible node tolerations
            self._prune_tolerations(
                generate_preemptible_tolerations(),
                tolerations_field_name=tolerations_field_name,
            )

            # purge affinity preemption related configuration
            self._prune_affinity_node_selector_requirement(
                generate_preemptible_node_selector_requirements(
                    NodeSelectorOperator.node_selector_op_in.value
                ),
                affinity_field_name=affinity_field_name,
            )
            # remove preemptible nodes constrain
            self._prune_node_selector(
                mlconf.get_preemptible_node_selector(),
                node_selector_field_name=node_selector_field_name,
            )

            # if tolerations are configured, simply pruning tolerations is sufficient because functions
            # cannot be scheduled without tolerations on tainted nodes.
            # however, if preemptible tolerations are not configured, we must use anti-affinity on preemptible nodes
            # to ensure that the function is not scheduled on the nodes.
            if not generate_preemptible_tolerations():
                # using a single term with potentially multiple expressions to ensure anti-affinity
                self._override_required_during_scheduling_ignored_during_execution(
                    k8s_client.V1NodeSelector(
                        node_selector_terms=generate_preemptible_nodes_anti_affinity_terms()
                    ),
                    affinity_field_name=affinity_field_name,
                )
        # enrich tolerations and override all node selector terms with preemptible node selector terms
        elif self_preemption_mode == PreemptionModes.constrain.value:
            # enrich with tolerations
            self._merge_tolerations(
                generate_preemptible_tolerations(),
                tolerations_field_name=tolerations_field_name,
            )

            # setting required_during_scheduling_ignored_during_execution
            # overriding other terms that have been set, and only setting terms for preemptible nodes
            # when having multiple terms, pod scheduling is succeeded if at least one term is satisfied
            self._override_required_during_scheduling_ignored_during_execution(
                k8s_client.V1NodeSelector(
                    node_selector_terms=generate_preemptible_nodes_affinity_terms()
                ),
                affinity_field_name=affinity_field_name,
            )
        # purge any affinity / anti-affinity preemption related configuration and enrich with preemptible tolerations
        elif self_preemption_mode == PreemptionModes.allow.value:

            # remove preemptible anti-affinity
            self._prune_affinity_node_selector_requirement(
                generate_preemptible_node_selector_requirements(
                    NodeSelectorOperator.node_selector_op_not_in.value
                ),
                affinity_field_name=affinity_field_name,
            )
            # remove preemptible affinity
            self._prune_affinity_node_selector_requirement(
                generate_preemptible_node_selector_requirements(
                    NodeSelectorOperator.node_selector_op_in.value
                ),
                affinity_field_name=affinity_field_name,
            )

            # remove preemptible nodes constrain
            self._prune_node_selector(
                mlconf.get_preemptible_node_selector(),
                node_selector_field_name=node_selector_field_name,
            )

            # enrich with tolerations
            self._merge_tolerations(
                generate_preemptible_tolerations(),
                tolerations_field_name=tolerations_field_name,
            )

        self._clear_affinity_if_initialized_but_empty(
            affinity_field_name=affinity_field_name
        )
        self._clear_tolerations_if_initialized_but_empty(
            tolerations_field_name=tolerations_field_name
        )

    def _clear_affinity_if_initialized_but_empty(self, affinity_field_name: str):
        self_affinity = getattr(self, affinity_field_name)
        if not getattr(self, affinity_field_name):
            setattr(self, affinity_field_name, None)
        elif (
            not self_affinity.node_affinity
            and not self_affinity.pod_affinity
            and not self_affinity.pod_anti_affinity
        ):
            setattr(self, affinity_field_name, None)

    def _clear_tolerations_if_initialized_but_empty(self, tolerations_field_name: str):
        if not getattr(self, tolerations_field_name):
            setattr(self, tolerations_field_name, None)

    def _merge_node_selector_term_to_node_affinity(
        self,
        node_selector_terms: typing.List[k8s_client.V1NodeSelectorTerm],
        affinity_field_name: str,
    ):
        if not node_selector_terms:
            return

        self._initialize_affinity(affinity_field_name)
        self._initialize_node_affinity(affinity_field_name)

        self_affinity = getattr(self, affinity_field_name)
        if (
            not self_affinity.node_affinity.required_during_scheduling_ignored_during_execution
        ):
            self_affinity.node_affinity.required_during_scheduling_ignored_during_execution = k8s_client.V1NodeSelector(
                node_selector_terms=node_selector_terms
            )
            return

        node_selector = (
            self_affinity.node_affinity.required_during_scheduling_ignored_during_execution
        )
        new_node_selector_terms = []

        for node_selector_term_to_add in node_selector_terms:
            to_add = True
            for node_selector_term in node_selector.node_selector_terms:
                if node_selector_term == node_selector_term_to_add:
                    to_add = False
                    break
            if to_add:
                new_node_selector_terms.append(node_selector_term_to_add)

        if new_node_selector_terms:
            node_selector.node_selector_terms += new_node_selector_terms

    def _initialize_affinity(self, affinity_field_name: str):
        if not getattr(self, affinity_field_name):
            setattr(self, affinity_field_name, k8s_client.V1Affinity())

    def _initialize_node_affinity(self, affinity_field_name: str):
        if not getattr(getattr(self, affinity_field_name), "node_affinity"):
            # self.affinity.node_affinity:
            getattr(
                self, affinity_field_name
            ).node_affinity = k8s_client.V1NodeAffinity()
            # self.affinity.node_affinity = k8s_client.V1NodeAffinity()

    def _prune_affinity_node_selector_requirement(
        self,
        node_selector_requirements: typing.List[k8s_client.V1NodeSelectorRequirement],
        affinity_field_name: str = "affinity",
    ):
        """
        Prunes given node selector requirements from affinity.
        We are only editing required_during_scheduling_ignored_during_execution because the scheduler can't schedule
        the pod unless the rule is met.
        :param node_selector_requirements:
        :return:
        """
        # both needs to exist to prune required affinity from spec affinity
        self_affinity = getattr(self, affinity_field_name)
        if not self_affinity or not node_selector_requirements:
            return
        if self_affinity.node_affinity:
            node_affinity: k8s_client.V1NodeAffinity = self_affinity.node_affinity

            new_required_during_scheduling_ignored_during_execution = None
            if node_affinity.required_during_scheduling_ignored_during_execution:
                node_selector: k8s_client.V1NodeSelector = (
                    node_affinity.required_during_scheduling_ignored_during_execution
                )
                new_node_selector_terms = (
                    self._prune_node_selector_requirements_from_node_selector_terms(
                        node_selector_terms=node_selector.node_selector_terms,
                        node_selector_requirements_to_prune=node_selector_requirements,
                    )
                )
                # check whether there are node selector terms to add to the new list of required terms
                if len(new_node_selector_terms) > 0:
                    new_required_during_scheduling_ignored_during_execution = (
                        k8s_client.V1NodeSelector(
                            node_selector_terms=new_node_selector_terms
                        )
                    )
            # if both preferred and new required are empty, clean node_affinity
            if (
                not node_affinity.preferred_during_scheduling_ignored_during_execution
                and not new_required_during_scheduling_ignored_during_execution
            ):
                setattr(self_affinity, "node_affinity", None)
                # self.affinity.node_affinity = None
                return

            self._initialize_affinity(affinity_field_name)
            self._initialize_node_affinity(affinity_field_name)

            self_affinity.node_affinity.required_during_scheduling_ignored_during_execution = (
                new_required_during_scheduling_ignored_during_execution
            )

    @staticmethod
    def _prune_node_selector_requirements_from_node_selector_terms(
        node_selector_terms: typing.List[k8s_client.V1NodeSelectorTerm],
        node_selector_requirements_to_prune: typing.List[
            k8s_client.V1NodeSelectorRequirement
        ],
    ) -> typing.List[k8s_client.V1NodeSelectorTerm]:
        """
        Goes over each expression in all the terms provided and removes the expressions if it matches
        one of the requirements provided to remove

        :return: New list of terms without the provided node selector requirements
        """
        new_node_selector_terms: typing.List[k8s_client.V1NodeSelectorTerm] = []
        for term in node_selector_terms:
            new_node_selector_requirements: typing.List[
                k8s_client.V1NodeSelectorRequirement
            ] = []
            for node_selector_requirement in term.match_expressions:
                to_prune = False
                # go over each requirement and check if matches the current expression
                for (
                    node_selector_requirement_to_prune
                ) in node_selector_requirements_to_prune:
                    if node_selector_requirement == node_selector_requirement_to_prune:
                        to_prune = True
                        # no need to keep going over the list provided for the current expression
                        break
                if not to_prune:
                    new_node_selector_requirements.append(node_selector_requirement)

            # check if there is something to add
            if len(new_node_selector_requirements) > 0 or term.match_fields:
                # Add new node selector terms without the matching expressions to prune
                new_node_selector_terms.append(
                    k8s_client.V1NodeSelectorTerm(
                        match_expressions=new_node_selector_requirements,
                        match_fields=term.match_fields,
                    )
                )
        return new_node_selector_terms

    def _prune_tolerations(
        self,
        tolerations: typing.List[k8s_client.V1Toleration],
        tolerations_field_name: str = "tolerations",
    ):
        """
        Prunes given tolerations from function spec
        :param tolerations: tolerations to prune
        """
        self_tolerations = getattr(self, tolerations_field_name)
        # both needs to exist to prune required tolerations from spec tolerations
        if not tolerations or not self_tolerations:
            return

        # generate a list of tolerations without tolerations to prune
        new_tolerations = []
        for toleration in self_tolerations:
            to_prune = False
            for toleration_to_delete in tolerations:
                if toleration == toleration_to_delete:
                    to_prune = True
                    # no need to keep going over the list provided for the current toleration
                    break
            if not to_prune:
                new_tolerations.append(toleration)

        # Set tolerations without tolerations to prune
        setattr(self, tolerations_field_name, new_tolerations)

    def _prune_node_selector(
        self,
        node_selector: typing.Dict[str, str],
        node_selector_field_name: str,
    ):
        """
        Prunes given node_selector key from function spec if their key and value are matching
        :param node_selector: node selectors to prune
        """
        self_node_selector = getattr(self, node_selector_field_name)
        # both needs to exists to prune required node_selector from the spec node selector
        if not node_selector or not self_node_selector:
            return

        for key, value in node_selector.items():
            if value:
                spec_value = self_node_selector.get(key)
                if spec_value and spec_value == value:
                    self_node_selector.pop(key)


class AutoMountType(str, Enum):
    none = "none"
    auto = "auto"
    v3io_credentials = "v3io_credentials"
    v3io_fuse = "v3io_fuse"
    pvc = "pvc"
    s3 = "s3"
    env = "env"

    @classmethod
    def _missing_(cls, value):
        if value:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid value for auto_mount_type - '{value}'"
            )
        return AutoMountType.default()

    @staticmethod
    def default():
        return AutoMountType.auto

    # Any modifier that configures a mount on a runtime should be included here. These modifiers, if applied to the
    # runtime, will suppress the auto-mount functionality.
    @classmethod
    def all_mount_modifiers(cls):
        return [
            mlrun.v3io_cred.__name__,
            mlrun.mount_v3io.__name__,
            mlrun.platforms.other.mount_pvc.__name__,
            mlrun.auto_mount.__name__,
            mlrun.platforms.mount_s3.__name__,
            mlrun.platforms.set_env_variables.__name__,
        ]

    @classmethod
    def is_auto_modifier(cls, modifier):
        # Check if modifier is one of the known mount modifiers. We need to use startswith since the modifier itself is
        # a nested function returned from the modifier function (such as 'v3io_cred.<locals>._use_v3io_cred')
        modifier_name = modifier.__qualname__
        return any(
            modifier_name.startswith(mount_modifier)
            for mount_modifier in AutoMountType.all_mount_modifiers()
        )

    @staticmethod
    def _get_auto_modifier():
        # If we're running on Iguazio - use v3io_cred
        if mlconf.igz_version != "":
            return mlrun.v3io_cred
        # Else, either pvc mount if it's configured or do nothing otherwise
        pvc_configured = (
            "MLRUN_PVC_MOUNT" in os.environ
            or "pvc_name" in mlconf.get_storage_auto_mount_params()
        )
        return mlrun.platforms.other.mount_pvc if pvc_configured else None

    def get_modifier(self):

        return {
            AutoMountType.none: None,
            AutoMountType.v3io_credentials: mlrun.v3io_cred,
            AutoMountType.v3io_fuse: mlrun.mount_v3io,
            AutoMountType.pvc: mlrun.platforms.other.mount_pvc,
            AutoMountType.auto: self._get_auto_modifier(),
            AutoMountType.s3: mlrun.platforms.mount_s3,
            AutoMountType.env: mlrun.platforms.set_env_variables,
        }[self]


class KubeResource(BaseRuntime):
    kind = "job"
    _is_nested = True

    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)
        self.verbose = False

    @property
    def spec(self) -> KubeResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", KubeResourceSpec)

    def to_dict(self, fields=None, exclude=None, strip=False):
        struct = super().to_dict(fields, exclude, strip=strip)
        api = k8s_client.ApiClient()
        struct = api.sanitize_for_serialization(struct)
        if strip:
            spec = struct["spec"]
            for attr in [
                "volumes",
                "volume_mounts",
                "driver_volume_mounts",
                "executor_volume_mounts",
            ]:
                if attr in spec:
                    del spec[attr]
            if "env" in spec and spec["env"]:
                for ev in spec["env"]:
                    if ev["name"].startswith("V3IO_"):
                        ev["value"] = ""
            # Reset this, since mounts and env variables were cleared.
            spec["disable_auto_mount"] = False
        return struct

    def apply(self, modify):
        """
        Apply a modifier to the runtime which is used to change the runtimes k8s object's spec.
        Modifiers can be either KFP modifiers or MLRun modifiers (which are compatible with KFP). All modifiers accept
        a `kfp.dsl.ContainerOp` object, apply some changes on its spec and return it so modifiers can be chained
        one after the other.

        :param modify: a modifier runnable object
        :return: the runtime (self) after the modifications
        """

        # Kubeflow pipeline have a hook to add the component to the DAG on ContainerOp init
        # we remove the hook to suppress kubeflow op registration and return it after the apply()
        old_op_handler = kfp.dsl._container_op._register_op_handler
        kfp.dsl._container_op._register_op_handler = lambda x: self.metadata.name
        cop = kfp.dsl.ContainerOp("name", "image")
        kfp.dsl._container_op._register_op_handler = old_op_handler

        return apply_kfp(modify, cop, self)

    def set_env_from_secret(self, name, secret=None, secret_key=None):
        """set pod environment var from secret"""
        secret_key = secret_key or name
        value_from = k8s_client.V1EnvVarSource(
            secret_key_ref=k8s_client.V1SecretKeySelector(name=secret, key=secret_key)
        )
        return self._set_env(name, value_from=value_from)

    def set_env(self, name, value=None, value_from=None):
        """set pod environment var from value"""
        if value is not None:
            return self._set_env(name, value=str(value))
        return self._set_env(name, value_from=value_from)

    def with_annotations(self, annotations: dict):
        """set a key/value annotations in the metadata of the pod"""
        for key, value in annotations.items():
            self.metadata.annotations[key] = str(value)
        return self

    def get_env(self, name, default=None):
        """Get the pod environment variable for the given name, if not found return the default
        If it's a scalar value, will return it, if the value is from source, return the k8s struct (V1EnvVarSource)"""
        for env_var in self.spec.env:
            if get_item_name(env_var) == name:
                value = get_item_name(env_var, "value")
                if value is not None:
                    return value
                return get_item_name(env_var, "value_from")
        return default

    def is_env_exists(self, name):
        """Check whether there is an environment variable define for the given key"""
        for env_var in self.spec.env:
            if get_item_name(env_var) == name:
                return True
        return False

    def _set_env(self, name, value=None, value_from=None):
        new_var = k8s_client.V1EnvVar(name=name, value=value, value_from=value_from)
        i = 0
        for v in self.spec.env:
            if get_item_name(v) == name:
                self.spec.env[i] = new_var
                return self
            i += 1
        self.spec.env.append(new_var)
        return self

    def set_envs(self, env_vars: dict = None, file_path: str = None):
        """set pod environment var from key/value dict or .env file

        :param env_vars:  dict with env key/values
        :param file_path: .env file with key=value lines
        """
        if (not env_vars and not file_path) or (env_vars and file_path):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "must specify env_vars OR file_path"
            )
        if file_path:
            env_vars = dotenv.dotenv_values(file_path)
            if None in env_vars.values():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "env file lines must be in the form key=value"
                )

        for name, value in env_vars.items():
            self.set_env(name, value)
        return self

    # TODO: Remove in 1.5.0
    @deprecated(
        version="1.3.0",
        reason="'Job gpus' will be removed in 1.5.0, use 'with_limits' instead",
        category=FutureWarning,
    )
    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        update_in(self.spec.resources, ["limits", gpu_type], gpus)

    def set_image_pull_configuration(
        self, image_pull_policy: str = None, image_pull_secret_name: str = None
    ):
        """
        Configure the image pull parameters for the runtime.

        :param image_pull_policy: The policy to use when pulling. One of `IfNotPresent`, `Always` or `Never`
        :param image_pull_secret_name: Name of a k8s secret containing image repository's authentication credentials
        """
        if image_pull_policy is not None:
            allowed_policies = ["Always", "IfNotPresent", "Never"]
            if image_pull_policy not in allowed_policies:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Image pull policy must be one of {allowed_policies}, got {image_pull_policy}"
                )
            self.spec.image_pull_policy = image_pull_policy
        if image_pull_secret_name is not None:
            self.spec.image_pull_secret = image_pull_secret_name

    def with_limits(
        self,
        mem: str = None,
        cpu: str = None,
        gpus: int = None,
        gpu_type: str = "nvidia.com/gpu",
        patch: bool = False,
    ):
        """
        set pod cpu/memory/gpu limits
        by default it overrides the whole limits section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec.with_limits(mem, cpu, gpus, gpu_type, patch=patch)

    def with_requests(self, mem: str = None, cpu: str = None, patch: bool = False):
        """
        set requested (desired) pod cpu/memory resources
        by default it overrides the whole requests section, if you wish to patch specific resources use `patch=True`.
        """
        self.spec.with_requests(mem, cpu, patch=patch)

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[k8s_client.V1Affinity] = None,
        tolerations: typing.Optional[typing.List[k8s_client.V1Toleration]] = None,
    ):
        """
        Enables to control on which k8s node the job will run

        :param node_name:       The name of the k8s node
        :param node_selector:   Label selector, only nodes with matching labels will be eligible to be picked
        :param affinity:        Expands the types of constraints you can express - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
                                for details
        :param tolerations:     Tolerations are applied to pods, and allow (but do not require) the pods to schedule
                                onto nodes with matching taints - see
                                https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration
                                for details

        """
        if node_name:
            self.spec.node_name = node_name
        if node_selector:
            self.spec.node_selector = node_selector
        if affinity:
            self.spec.affinity = affinity
        if tolerations is not None:
            self.spec.tolerations = tolerations

    def with_priority_class(self, name: typing.Optional[str] = None):
        """
        Enables to control the priority of the pod
        If not passed - will default to mlrun.mlconf.default_function_priority_class_name

        :param name:       The name of the priority class
        """
        if name is None:
            name = mlconf.default_function_priority_class_name
        valid_priority_class_names = self.list_valid_priority_class_names()
        if name not in valid_priority_class_names:
            message = "Priority class name not in available priority class names"
            logger.warning(
                message,
                priority_class_name=name,
                valid_priority_class_names=valid_priority_class_names,
            )
            raise mlrun.errors.MLRunInvalidArgumentError(message)
        self.spec.priority_class_name = name

    def with_preemption_mode(self, mode: typing.Union[PreemptionModes, str]):
        """
        Preemption mode controls whether pods can be scheduled on preemptible nodes.
        Tolerations, node selector, and affinity are populated on preemptible nodes corresponding to the function spec.

        The supported modes are:

        * **allow** - The function can be scheduled on preemptible nodes
        * **constrain** - The function can only run on preemptible nodes
        * **prevent** - The function cannot be scheduled on preemptible nodes
        * **none** - No preemptible configuration will be applied on the function

        The default preemption mode is configurable in mlrun.mlconf.function_defaults.preemption_mode,
        by default it's set to **prevent**

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.api.schemas.PreemptionModes`
        """
        preemptible_mode = PreemptionModes(mode)
        self.spec.preemption_mode = preemptible_mode.value

    def with_security_context(self, security_context: k8s_client.V1SecurityContext):
        """
        Set security context for the pod.
        For Iguazio we handle security context internally -
        see mlrun.api.schemas.function.SecurityContextEnrichmentModes

        Example:

            from kubernetes import client as k8s_client

            security_context = k8s_client.V1SecurityContext(
                        run_as_user=1000,
                        run_as_group=3000,
                    )
            function.with_security_context(security_context)

        More info:
        https://kubernetes.io/docs/tasks/configure-pod-container/security-context/#set-the-security-context-for-a-pod

        :param security_context:         The security context for the pod
        """
        if (
            mlrun.mlconf.function.spec.security_context.enrichment_mode
            != SecurityContextEnrichmentModes.disabled.value
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Security context is handled internally when enrichment mode is not disabled"
            )
        self.spec.security_context = security_context

    def list_valid_priority_class_names(self):
        return mlconf.get_valid_function_priority_class_names()

    def get_default_priority_class_name(self):
        return mlconf.default_function_priority_class_name

    def _get_meta(self, runobj, unique=False):
        namespace = self._get_k8s().resolve_namespace()

        labels = get_resource_labels(self, runobj, runobj.spec.scrape_metrics)
        new_meta = k8s_client.V1ObjectMeta(
            namespace=namespace,
            annotations=self.metadata.annotations or runobj.metadata.annotations,
            labels=labels,
        )

        name = runobj.metadata.name or "mlrun"
        norm_name = f"{normalize_name(name)}-"
        if unique:
            norm_name += uuid.uuid4().hex[:8]
            new_meta.name = norm_name
            runobj.set_label("mlrun/job", norm_name)
        else:
            new_meta.generate_name = norm_name
        return new_meta

    def _add_secrets_to_spec_before_running(self, runobj=None, project=None):
        if self._secrets:
            if self._secrets.has_vault_source():
                self._add_vault_params_to_spec(runobj=runobj, project=project)
            if self._secrets.has_azure_vault_source():
                self._add_azure_vault_params_to_spec(
                    self._secrets.get_azure_vault_k8s_secret()
                )
            self._add_k8s_secrets_to_spec(
                self._secrets.get_k8s_secrets(), runobj=runobj, project=project
            )
        else:
            self._add_k8s_secrets_to_spec(None, runobj=runobj, project=project)

    def _add_azure_vault_params_to_spec(self, k8s_secret_name=None):
        secret_name = (
            k8s_secret_name or mlconf.secret_stores.azure_vault.default_secret_name
        )
        if not secret_name:
            logger.warning(
                "No k8s secret provided. Azure key vault will not be available"
            )
            return

        # We cannot use expanduser() here, since the user in question is the user running in the pod
        # itself (which is root) and not where this code is running. That's why this hacky replacement is needed.
        secret_path = mlconf.secret_stores.azure_vault.secret_path.replace("~", "/root")
        volumes = [
            {
                "name": "azure-vault-secret",
                "secret": {"defaultMode": 420, "secretName": secret_name},
            }
        ]
        volume_mounts = [{"name": "azure-vault-secret", "mountPath": secret_path}]
        self.spec.update_vols_and_mounts(volumes, volume_mounts)

    def _add_k8s_secrets_to_spec(
        self,
        secrets,
        runobj=None,
        project=None,
        encode_key_names=True,
    ):
        # Check if we need to add the keys of a global secret. Global secrets are intentionally added before
        # project secrets, to allow project secret keys to override them
        global_secret_name = (
            mlconf.secret_stores.kubernetes.global_function_env_secret_name
        )
        if mlrun.config.is_running_as_api() and global_secret_name:
            global_secrets = self._get_k8s().get_secret_data(global_secret_name)
            for key, value in global_secrets.items():
                env_var_name = (
                    SecretsStore.k8s_env_variable_name_for_secret(key)
                    if encode_key_names
                    else key
                )
                self.set_env_from_secret(env_var_name, global_secret_name, key)

        # the secrets param may be an empty dictionary (asking for all secrets of that project) -
        # it's a different case than None (not asking for project secrets at all).
        if (
            secrets is None
            and not mlconf.secret_stores.kubernetes.auto_add_project_secrets
        ):
            return

        project_name = project or runobj.metadata.project
        if project_name is None:
            logger.warning("No project provided. Cannot add k8s secrets")
            return

        secret_name = self._get_k8s().get_project_secret_name(project_name)
        # Not utilizing the same functionality from the Secrets crud object because this code also runs client-side
        # in the nuclio remote-dashboard flow, which causes dependency problems.
        existing_secret_keys = self._get_k8s().get_project_secret_keys(
            project_name, filter_internal=True
        )

        # If no secrets were passed or auto-adding all secrets, we need all existing keys
        if not secrets:
            secrets = {
                key: SecretsStore.k8s_env_variable_name_for_secret(key)
                if encode_key_names
                else key
                for key in existing_secret_keys
            }

        for key, env_var_name in secrets.items():
            if key in existing_secret_keys:
                self.set_env_from_secret(env_var_name, secret_name, key)

        # Keep a list of the variables that relate to secrets, so that the MLRun context (when using nuclio:mlrun)
        # can be initialized with those env variables as secrets
        if not encode_key_names and secrets.keys():
            self.set_env("MLRUN_PROJECT_SECRETS_LIST", ",".join(secrets.keys()))

    def _add_vault_params_to_spec(self, runobj=None, project=None):
        project_name = project or runobj.metadata.project
        if project_name is None:
            logger.warning("No project provided. Cannot add vault parameters")
            return

        service_account_name = (
            mlconf.secret_stores.vault.project_service_account_name.format(
                project=project_name
            )
        )

        project_vault_secret_name = self._get_k8s().get_project_vault_secret_name(
            project_name, service_account_name
        )
        if project_vault_secret_name is None:
            logger.info(f"No vault secret associated with project {project_name}")
            return

        volumes = [
            {
                "name": "vault-secret",
                "secret": {"defaultMode": 420, "secretName": project_vault_secret_name},
            }
        ]
        # We cannot use expanduser() here, since the user in question is the user running in the pod
        # itself (which is root) and not where this code is running. That's why this hacky replacement is needed.
        token_path = mlconf.secret_stores.vault.token_path.replace("~", "/root")

        volume_mounts = [{"name": "vault-secret", "mountPath": token_path}]

        self.spec.update_vols_and_mounts(volumes, volume_mounts)
        self.spec.env.append(
            {
                "name": "MLRUN_SECRET_STORES__VAULT__ROLE",
                "value": f"project:{project_name}",
            }
        )
        # In case remote URL is different than local URL, use it. Else, use the local URL
        vault_url = mlconf.secret_stores.vault.remote_url
        if vault_url == "":
            vault_url = mlconf.secret_stores.vault.url

        self.spec.env.append(
            {"name": "MLRUN_SECRET_STORES__VAULT__URL", "value": vault_url}
        )

    def try_auto_mount_based_on_config(self, override_params=None):
        if self.spec.disable_auto_mount:
            logger.debug(
                "Mount already applied or auto-mount manually disabled - not performing auto-mount"
            )
            return

        auto_mount_type = AutoMountType(mlconf.storage.auto_mount_type)
        modifier = auto_mount_type.get_modifier()
        if not modifier:
            logger.debug(
                "Auto mount disabled due to user selection (auto_mount_type=none)"
            )
            return

        mount_params_dict = mlconf.get_storage_auto_mount_params()
        override_params = override_params or {}
        for key, value in override_params.items():
            mount_params_dict[key] = value

        mount_params_dict = _filter_modifier_params(modifier, mount_params_dict)

        self.apply(modifier(**mount_params_dict))

    def validate_and_enrich_service_account(
        self, allowed_service_accounts, default_service_account
    ):
        if not self.spec.service_account:
            if default_service_account:
                self.spec.service_account = default_service_account
                logger.info(
                    f"Setting default service account to function: {default_service_account}"
                )

        self.spec.validate_service_account(allowed_service_accounts)


def kube_resource_spec_to_pod_spec(
    kube_resource_spec: KubeResourceSpec, container: k8s_client.V1Container
):
    return k8s_client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        volumes=kube_resource_spec.volumes,
        service_account=kube_resource_spec.service_account,
        node_name=kube_resource_spec.node_name,
        node_selector=kube_resource_spec.node_selector,
        affinity=kube_resource_spec.affinity,
        priority_class_name=kube_resource_spec.priority_class_name
        if len(mlconf.get_valid_function_priority_class_names())
        else None,
        tolerations=kube_resource_spec.tolerations,
        security_context=kube_resource_spec.security_context,
    )


def _resolve_if_type_sanitized(attribute_name, attribute):
    attribute_config = sanitized_attributes[attribute_name]
    # heuristic - if one of the keys contains _ as part of the dict it means to_dict on the kubernetes
    # object performed, there's nothing we can do at that point to transform it to the sanitized version
    for key in attribute.keys():
        if "_" in key:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"{attribute_name} must be instance of kubernetes {attribute_config.get('attribute_type_name')} class "
                f"but contains not sanitized key: {key}"
            )

    # then it's already the sanitized version
    return attribute


def transform_attribute_to_k8s_class_instance(
    attribute_name, attribute, is_sub_attr: bool = False
):
    if attribute_name not in sanitized_attributes:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"{attribute_name} isn't in the available sanitized attributes"
        )
    attribute_config = sanitized_attributes[attribute_name]
    # initialize empty attribute type
    if attribute is None:
        return None
    if isinstance(attribute, dict):
        if _resolve_if_type_sanitized(attribute_name, attribute):
            api = k8s_client.ApiClient()
            # not ideal to use their private method, but looks like that's the only option
            # Taken from https://github.com/kubernetes-client/python/issues/977
            attribute_type = attribute_config["attribute_type"]
            if attribute_config["contains_many"]:
                attribute_type = attribute_config["sub_attribute_type"]
            attribute = api._ApiClient__deserialize(attribute, attribute_type)

    elif isinstance(attribute, list):
        attribute_instance = []
        for sub_attr in attribute:
            if not isinstance(sub_attr, dict):
                return attribute
            attribute_instance.append(
                transform_attribute_to_k8s_class_instance(
                    attribute_name, sub_attr, is_sub_attr=True
                )
            )
        attribute = attribute_instance
    # if user have set one attribute but its part of an attribute that contains many then return inside a list
    if (
        not is_sub_attr
        and attribute_config["contains_many"]
        and isinstance(attribute, attribute_config["sub_attribute_type"])
    ):
        # initialize attribute instance and add attribute to it,
        # mainly done when attribute is a list but user defines only sets the attribute not in the list
        attribute_instance = attribute_config["attribute_type"]()
        attribute_instance.append(attribute)
        return attribute_instance
    return attribute


def get_sanitized_attribute(spec, attribute_name: str):
    """
    When using methods like to_dict() on kubernetes class instances we're getting the attributes in snake_case
    Which is ok if we're using the kubernetes python package but not if for example we're creating CRDs that we
    apply directly. For that we need the sanitized (CamelCase) version.
    """
    attribute = getattr(spec, attribute_name)
    if attribute_name not in sanitized_attributes:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"{attribute_name} isn't in the available sanitized attributes"
        )
    attribute_config = sanitized_attributes[attribute_name]
    if not attribute:
        return attribute_config["not_sanitized_class"]()

    # check if attribute of type dict, and then check if type is sanitized
    if isinstance(attribute, dict):
        if attribute_config["not_sanitized_class"] != dict:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"expected to to be of type {attribute_config.get('not_sanitized_class')} but got dict"
            )
        if _resolve_if_type_sanitized(attribute_name, attribute):
            return attribute

    elif isinstance(attribute, list) and not isinstance(
        attribute[0], attribute_config["sub_attribute_type"]
    ):
        if attribute_config["not_sanitized_class"] != list:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"expected to to be of type {attribute_config.get('not_sanitized_class')} but got list"
            )
        if _resolve_if_type_sanitized(attribute_name, attribute[0]):
            return attribute

    api = k8s_client.ApiClient()
    return api.sanitize_for_serialization(attribute)


def _filter_modifier_params(modifier, params):
    # Make sure we only pass parameters that are accepted by the modifier.
    modifier_params = inspect.signature(modifier).parameters

    # If kwargs are supported by the modifier, we don't filter.
    if any(param.kind == param.VAR_KEYWORD for param in modifier_params.values()):
        return params

    param_names = modifier_params.keys()
    filtered_params = {}
    for key, value in params.items():
        if key in param_names:
            filtered_params[key] = value
        else:
            logger.warning(
                "Auto mount parameter not supported by modifier, filtered out",
                modifier=modifier.__name__,
                param=key,
            )
    return filtered_params
