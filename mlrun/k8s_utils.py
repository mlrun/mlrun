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
import copy
import re
import typing
import warnings

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


def verify_label_key(key: str, allow_k8s_prefix: bool = False):
    """
    Verify that the label key is valid for Kubernetes.
    Refer to https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set
    """
    if not key:
        raise mlrun.errors.MLRunInvalidArgumentError("label key cannot be empty")

    prefix = ""
    parts = key.split("/")
    if len(parts) == 1:
        name = parts[0]
    elif len(parts) == 2:
        prefix, name = parts
        if len(name) == 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Label key name cannot be empty when a prefix is set"
            )
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

    # Allow the use of Kubernetes reserved prefixes ('k8s.io/' or 'kubernetes.io/')
    # only when setting node selectors, not when adding new labels.
    if not allow_k8s_prefix and prefix in {"k8s.io", "kubernetes.io"}:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Labels cannot start with 'k8s.io/' or 'kubernetes.io/'"
        )


def verify_label_value(value, label_key):
    mlrun.utils.helpers.verify_field_regex(
        f"project.metadata.labels.'{label_key}'",
        value,
        mlrun.utils.regex.label_value,
    )


def validate_node_selectors(
    node_selectors: dict[str, str], raise_on_error: bool = True
) -> bool:
    """
    Ensures that user-defined node selectors adhere to Kubernetes label standards:
    - Validates that each key conforms to Kubernetes naming conventions, with specific rules for name and prefix.
    - Ensures values comply with Kubernetes label value rules.
    - If raise_on_error is True, raises errors for invalid selectors.
    - If raise_on_error is False, logs warnings for invalid selectors.
    """

    # Helper function for handling errors or warnings
    def handle_invalid(message):
        if raise_on_error:
            raise
        else:
            warnings.warn(
                f"{message}\n"
                f"The node selector you’ve set does not meet the validation rules for the current Kubernetes version. "
                f"Please note that invalid node selectors may cause issues with function scheduling."
            )

    node_selectors = node_selectors or {}
    for key, value in node_selectors.items():
        try:
            verify_label_key(key, allow_k8s_prefix=True)
            verify_label_value(value, label_key=key)
        except mlrun.errors.MLRunInvalidArgumentError as err:
            # An error or warning is raised by handle_invalid due to validation failure.
            # Returning False indicates validation failed, allowing us to exit the function.
            handle_invalid(str(err))
            return False
    return True


def enrich_preemption_mode(
    preemption_mode: typing.Optional[str],
    node_selector: dict[str, str],
    tolerations: list[kubernetes.client.V1Toleration],
    affinity: typing.Optional[kubernetes.client.V1Affinity],
) -> tuple[
    dict[str, str],
    list[kubernetes.client.V1Toleration],
    typing.Optional[kubernetes.client.V1Affinity],
]:
    """
    Enriches a pod spec's scheduling configuration (node selector, tolerations, affinity)
    based on the provided preemption mode.

    If no preemptible node configuration is defined in the system, or the mode is `none`,
    the original values are returned unchanged.

    Modes:
        - allow: Adds tolerations, removes preemption constraints.
        - constrain: Requires preemptible node affinity and adds tolerations.
        - prevent: Enforces scheduling on non-preemptible nodes using taints or anti-affinity.
        - none: No enrichment is applied.
    """
    if (
        not mlconfig.is_preemption_nodes_configured()
        or preemption_mode == mlrun.common.schemas.PreemptionModes.none.value
    ):
        return node_selector, tolerations, affinity

    if not preemption_mode:
        preemption_mode = mlconfig.function_defaults.preemption_mode
        mlrun.utils.logger.debug(
            "No preemption mode provided, using default",
            default_preemption_mode=preemption_mode,
        )

    enriched_node_selector = copy.deepcopy(node_selector or {})
    enriched_tolerations = copy.deepcopy(tolerations or [])
    enriched_affinity = copy.deepcopy(affinity)
    preemptible_tolerations = generate_preemptible_tolerations()

    if handler := _get_mode_handler(preemption_mode):
        enriched_node_selector, enriched_tolerations, enriched_affinity = handler(
            enriched_node_selector,
            enriched_tolerations,
            enriched_affinity,
            preemptible_tolerations,
        )

    return (
        enriched_node_selector,
        enriched_tolerations,
        _prune_empty_affinity(enriched_affinity),
    )


def _get_mode_handler(mode: str):
    return {
        mlrun.common.schemas.PreemptionModes.prevent: _handle_prevent_mode,
        mlrun.common.schemas.PreemptionModes.constrain: _handle_constrain_mode,
        mlrun.common.schemas.PreemptionModes.allow: _handle_allow_mode,
    }.get(mode)


def _handle_prevent_mode(
    node_selector: dict[str, str],
    tolerations: list[kubernetes.client.V1Toleration],
    affinity: typing.Optional[kubernetes.client.V1Affinity],
    preemptible_tolerations: list[kubernetes.client.V1Toleration],
) -> tuple[
    dict[str, str],
    list[kubernetes.client.V1Toleration],
    typing.Optional[kubernetes.client.V1Affinity],
]:
    # Ensure no preemptible node tolerations
    tolerations = [t for t in tolerations if t not in preemptible_tolerations]

    # Purge affinity preemption-related configuration
    affinity = _prune_affinity_node_selector_requirement(
        generate_preemptible_node_selector_requirements(
            mlrun.common.schemas.NodeSelectorOperator.node_selector_op_in.value
        ),
        affinity=affinity,
    )

    # Remove preemptible nodes constraint
    node_selector = _prune_node_selector(
        mlconfig.get_preemptible_node_selector(),
        enriched_node_selector=node_selector,
    )

    # Use anti-affinity only if no tolerations configured
    if not preemptible_tolerations:
        affinity = _override_required_during_scheduling_ignored_during_execution(
            kubernetes.client.V1NodeSelector(
                node_selector_terms=generate_preemptible_nodes_anti_affinity_terms()
            ),
            affinity,
        )

    return node_selector, tolerations, affinity


def _handle_constrain_mode(
    node_selector: dict[str, str],
    tolerations: list[kubernetes.client.V1Toleration],
    affinity: typing.Optional[kubernetes.client.V1Affinity],
    preemptible_tolerations: list[kubernetes.client.V1Toleration],
) -> tuple[
    dict[str, str],
    list[kubernetes.client.V1Toleration],
    typing.Optional[kubernetes.client.V1Affinity],
]:
    tolerations = _merge_tolerations(tolerations, preemptible_tolerations)

    affinity = _override_required_during_scheduling_ignored_during_execution(
        kubernetes.client.V1NodeSelector(
            node_selector_terms=generate_preemptible_nodes_affinity_terms()
        ),
        affinity=affinity,
    )

    return node_selector, tolerations, affinity


def _handle_allow_mode(
    node_selector: dict[str, str],
    tolerations: list[kubernetes.client.V1Toleration],
    affinity: typing.Optional[kubernetes.client.V1Affinity],
    preemptible_tolerations: list[kubernetes.client.V1Toleration],
) -> tuple[
    dict[str, str],
    list[kubernetes.client.V1Toleration],
    typing.Optional[kubernetes.client.V1Affinity],
]:
    for op in [
        mlrun.common.schemas.NodeSelectorOperator.node_selector_op_not_in.value,
        mlrun.common.schemas.NodeSelectorOperator.node_selector_op_in.value,
    ]:
        affinity = _prune_affinity_node_selector_requirement(
            generate_preemptible_node_selector_requirements(op),
            affinity=affinity,
        )

    node_selector = _prune_node_selector(
        mlconfig.get_preemptible_node_selector(),
        enriched_node_selector=node_selector,
    )

    tolerations = _merge_tolerations(tolerations, preemptible_tolerations)
    return node_selector, tolerations, affinity


def _merge_tolerations(
    existing: list[kubernetes.client.V1Toleration],
    to_add: list[kubernetes.client.V1Toleration],
) -> list[kubernetes.client.V1Toleration]:
    for toleration in to_add:
        if toleration not in existing:
            existing.append(toleration)
    return existing


def _prune_node_selector(
    node_selector: dict[str, str],
    enriched_node_selector: dict[str, str],
):
    """
    Prunes given node_selector key from function spec if their key and value are matching
    :param node_selector: node selectors to prune
    """
    # both needs to exists to prune required node_selector from the spec node selector
    if not node_selector or not enriched_node_selector:
        return

    mlrun.utils.logger.debug("Pruning node selectors", node_selector=node_selector)
    return {
        key: value
        for key, value in enriched_node_selector.items()
        if node_selector.get(key) != value
    }


def _prune_affinity_node_selector_requirement(
    node_selector_requirements: list[kubernetes.client.V1NodeSelectorRequirement],
    affinity: typing.Optional[kubernetes.client.V1Affinity],
):
    """
    Prunes given node selector requirements from affinity.
    We are only editing required_during_scheduling_ignored_during_execution because the scheduler can't schedule
    the pod unless the rule is met.
    :param node_selector_requirements:
    :return:
    """
    # both needs to exist to prune required affinity from spec affinity
    if not affinity or not node_selector_requirements:
        return
    if affinity.node_affinity:
        node_affinity: kubernetes.client.V1NodeAffinity = affinity.node_affinity

        new_required_during_scheduling_ignored_during_execution = None
        if node_affinity.required_during_scheduling_ignored_during_execution:
            node_selector: kubernetes.client.V1NodeSelector = (
                node_affinity.required_during_scheduling_ignored_during_execution
            )
            new_node_selector_terms = (
                _prune_node_selector_requirements_from_node_selector_terms(
                    node_selector_terms=node_selector.node_selector_terms,
                    requirements_to_prune=node_selector_requirements,
                )
            )
            # check whether there are node selector terms to add to the new list of required terms
            if new_node_selector_terms:
                new_required_during_scheduling_ignored_during_execution = (
                    kubernetes.client.V1NodeSelector(
                        node_selector_terms=new_node_selector_terms
                    )
                )
        # if both preferred and new required are empty, clean node_affinity
        if (
            not node_affinity.preferred_during_scheduling_ignored_during_execution
            and not new_required_during_scheduling_ignored_during_execution
        ):
            affinity.node_affinity = None
            return

        _initialize_affinity(affinity=affinity)
        _initialize_node_affinity(affinity=affinity)

        affinity.node_affinity.required_during_scheduling_ignored_during_execution = (
            new_required_during_scheduling_ignored_during_execution
        )
        return affinity


def _prune_node_selector_requirements_from_node_selector_terms(
    node_selector_terms: list[kubernetes.client.V1NodeSelectorTerm],
    requirements_to_prune: list[kubernetes.client.V1NodeSelectorRequirement],
) -> list[kubernetes.client.V1NodeSelectorTerm]:
    """
    Removes matching node selector requirements from the given list of node selector terms.

    Each term may contain multiple match expressions. This function iterates over each expression,
    and removes any that exactly match one of the requirements provided.

    :param node_selector_terms: List of V1NodeSelectorTerm objects to be processed.
    :param requirements_to_prune: List of V1NodeSelectorRequirement objects to remove.
    :return: A new list of V1NodeSelectorTerm objects with the specified requirements pruned.
    """
    pruned_terms = []

    for term in node_selector_terms:
        remaining_requirements = [
            expr
            for expr in term.match_expressions or []
            if expr not in requirements_to_prune
        ]

        # Only add term if there are remaining match expressions or match fields
        if remaining_requirements or term.match_fields:
            pruned_terms.append(
                kubernetes.client.V1NodeSelectorTerm(
                    match_expressions=remaining_requirements,
                    match_fields=term.match_fields,
                )
            )

    return pruned_terms


def _override_required_during_scheduling_ignored_during_execution(
    node_selector: kubernetes.client.V1NodeSelector,
    affinity: typing.Optional[kubernetes.client.V1Affinity],
):
    affinity = _initialize_affinity(affinity)
    affinity = _initialize_node_affinity(affinity)
    affinity.node_affinity.required_during_scheduling_ignored_during_execution = (
        node_selector
    )
    return affinity


def _initialize_affinity(
    affinity: typing.Optional[kubernetes.client.V1Affinity],
) -> kubernetes.client.V1Affinity:
    return affinity or kubernetes.client.V1Affinity()


def _initialize_node_affinity(
    affinity: typing.Optional[kubernetes.client.V1Affinity],
) -> kubernetes.client.V1Affinity:
    affinity = affinity or kubernetes.client.V1Affinity()
    affinity.node_affinity = (
        affinity.node_affinity or kubernetes.client.V1NodeAffinity()
    )
    return affinity


def _prune_empty_affinity(
    affinity: typing.Optional[kubernetes.client.V1Affinity],
) -> typing.Optional[kubernetes.client.V1Affinity]:
    """
    Return None if the given affinity object has no meaningful constraints.

    Keeps the affinity object only if it contains:
    - Any pod affinity or pod anti-affinity
    - Preferred node affinity
    - Required node affinity with at least one match expression or match field
    """
    if not affinity:
        return None

    node_affinity = affinity.node_affinity
    pod_affinity = affinity.pod_affinity
    pod_anti_affinity = affinity.pod_anti_affinity

    # If any pod affinity exists, keep the object
    if pod_affinity or pod_anti_affinity:
        return affinity

    # If node affinity exists, check if it has any meaningful content
    if node_affinity:
        required = node_affinity.required_during_scheduling_ignored_during_execution
        preferred = node_affinity.preferred_during_scheduling_ignored_during_execution

        if preferred:
            return affinity

        if required and required.node_selector_terms:
            for term in required.node_selector_terms:
                if term.match_expressions or term.match_fields:
                    return affinity  # at least one term has meaningful constraints

    # At this point, none of the affinity sections contain meaningful constraints,
    # so the affinity object is effectively empty and can be safely discarded.
    return None
