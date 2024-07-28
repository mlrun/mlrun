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
import re

import kubernetes.client

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.regex

from .config import config as mlconfig

_running_inside_kubernetes_cluster = None


def is_running_inside_kubernetes_cluster():
    global _running_inside_kubernetes_cluster
    if _running_inside_kubernetes_cluster is None:
        try:
            kubernetes.config.load_incluster_config()
            _running_inside_kubernetes_cluster = True
        except kubernetes.config.ConfigException:
            _running_inside_kubernetes_cluster = False
    return _running_inside_kubernetes_cluster


def generate_preemptible_node_selector_requirements(
    node_selector_operator: str,
) -> list[kubernetes.client.V1NodeSelectorRequirement]:
    """
    Generate node selector requirements based on the pre-configured node selector of the preemptible nodes.
    node selector operator represents a key's relationship to a set of values.
    Valid operators are listed in :py:class:`~mlrun.common.schemas.NodeSelectorOperator`
    :param node_selector_operator: The operator of V1NodeSelectorRequirement
    :return: List[V1NodeSelectorRequirement]
    """
    match_expressions = []
    for (
        node_selector_key,
        node_selector_value,
    ) in mlconfig.get_preemptible_node_selector().items():
        match_expressions.append(
            kubernetes.client.V1NodeSelectorRequirement(
                key=node_selector_key,
                operator=node_selector_operator,
                values=[node_selector_value],
            )
        )
    return match_expressions


def generate_preemptible_nodes_anti_affinity_terms() -> (
    list[kubernetes.client.V1NodeSelectorTerm]
):
    """
    Generate node selector term containing anti-affinity expressions based on the
    pre-configured node selector of the preemptible nodes.
    Use for purpose of scheduling on node only if all match_expressions are satisfied.
    This function uses a single term with potentially multiple expressions to ensure anti affinity.
    https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
    :return: List contains one nodeSelectorTerm with multiple expressions.
    """
    # compile affinities with operator NotIn to make sure pods are not running on preemptible nodes.
    node_selector_requirements = generate_preemptible_node_selector_requirements(
        mlrun.common.schemas.NodeSelectorOperator.node_selector_op_not_in.value
    )
    return [
        kubernetes.client.V1NodeSelectorTerm(
            match_expressions=node_selector_requirements,
        )
    ]


def generate_preemptible_nodes_affinity_terms() -> (
    list[kubernetes.client.V1NodeSelectorTerm]
):
    """
    Use for purpose of scheduling on node having at least one of the node selectors.
    When specifying multiple nodeSelectorTerms associated with nodeAffinity types,
    then the pod can be scheduled onto a node if at least one of the nodeSelectorTerms can be satisfied.
    :return: List of nodeSelectorTerms associated with the preemptible nodes.
    """
    node_selector_terms = []

    # compile affinities with operator In so pods could schedule on at least one of the preemptible nodes.
    node_selector_requirements = generate_preemptible_node_selector_requirements(
        mlrun.common.schemas.NodeSelectorOperator.node_selector_op_in.value
    )
    for expression in node_selector_requirements:
        node_selector_terms.append(
            kubernetes.client.V1NodeSelectorTerm(match_expressions=[expression])
        )
    return node_selector_terms


def generate_preemptible_tolerations() -> list[kubernetes.client.V1Toleration]:
    tolerations = mlconfig.get_preemptible_tolerations()

    toleration_objects = []
    for toleration in tolerations:
        toleration_objects.append(
            kubernetes.client.V1Toleration(
                effect=toleration.get("effect", None),
                key=toleration.get("key", None),
                value=toleration.get("value", None),
                operator=toleration.get("operator", None),
                toleration_seconds=toleration.get("toleration_seconds", None)
                or toleration.get("tolerationSeconds", None),
            )
        )
    return toleration_objects


def sanitize_label_value(value: str) -> str:
    """
    Kubernetes label values must be sanitized before they're sent to the API
    Refer to https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set

    :param value: arbitrary string that needs to sanitized for usage on k8s labels
    :return:      string fully compliant with k8s label value expectations
    """
    return re.sub(r"([^a-zA-Z0-9_.-]|^[^a-zA-Z0-9]|[^a-zA-Z0-9]$)", "-", value[:63])


def verify_label_key(key: str):
    """
    Verify that the label key is valid for Kubernetes.
    Refer to https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set
    """
    if not key:
        raise mlrun.errors.MLRunInvalidArgumentError("label key cannot be empty")

    parts = key.split("/")
    if len(parts) == 1:
        name = parts[0]
    elif len(parts) == 2:
        prefix, name = parts
        if len(prefix) == 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Label key prefix cannot be empty"
            )

        # prefix must adhere dns_1123_subdomain
        mlrun.utils.helpers.verify_field_regex(
            f"Project.metadata.labels.'{key}'",
            prefix,
            mlrun.utils.regex.dns_1123_subdomain,
        )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Label key can only contain one '/'"
        )

    mlrun.utils.helpers.verify_field_regex(
        f"project.metadata.labels.'{key}'",
        name,
        mlrun.utils.regex.k8s_character_limit,
    )
    mlrun.utils.helpers.verify_field_regex(
        f"project.metadata.labels.'{key}'",
        name,
        mlrun.utils.regex.qualified_name,
    )

    if key.startswith("k8s.io/") or key.startswith("kubernetes.io/"):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Labels cannot start with 'k8s.io/' or 'kubernetes.io/'"
        )


def verify_label_value(value, label_key):
    mlrun.utils.helpers.verify_field_regex(
        f"project.metadata.labels.'{label_key}'",
        value,
        mlrun.utils.regex.label_value,
    )
