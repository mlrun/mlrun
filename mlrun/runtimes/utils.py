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
import enum
import getpass
import hashlib
import json
import os
import re
from io import StringIO
from sys import stderr
from typing import Optional

import pandas as pd

import mlrun
import mlrun.common.constants
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.utils.regex
from mlrun.artifacts import TableArtifact
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.frameworks.parallel_coordinates import gen_pcp_plot
from mlrun.runtimes.generators import selector
from mlrun.utils import get_in, helpers, logger, verify_field_regex


class RunError(Exception):
    pass


class _ContextStore:
    def __init__(self):
        self._context = None

    def get(self):
        return self._context

    def set(self, context):
        self._context = context


global_context = _ContextStore()


def resolve_spark_operator_version():
    try:
        regex = re.compile("spark-([23])")
        return int(regex.findall(config.spark_operator_version)[0])
    except Exception:
        raise ValueError("Failed to resolve spark operator's version")


def calc_hash(func, tag=""):
    # remove tag, hash, date from calculation
    tag = tag or func.metadata.tag
    status = func.status
    func.metadata.tag = ""
    func.metadata.hash = ""
    func.status = None
    func.metadata.updated = None

    data = json.dumps(func.to_dict(), sort_keys=True).encode()
    h = hashlib.sha1()
    h.update(data)
    hashkey = h.hexdigest()
    func.metadata.tag = tag
    func.metadata.hash = hashkey
    func.status = status
    return hashkey


def log_std(db, runobj, out, err="", skip=False, show=True, silent=False):
    if out:
        iteration = runobj.metadata.iteration
        if iteration:
            line = "> " + "-" * 15 + f" Iteration: ({iteration}) " + "-" * 15 + "\n"
            out = line + out
        if show:
            print(out, flush=True)
        if db and not skip:
            uid = runobj.metadata.uid
            project = runobj.metadata.project or ""
            db.store_log(uid, project, out.encode(), append=True)
    if err:
        logger.error(f"Exec error - {err_to_str(err)}")
        print(err, file=stderr)
        if not silent:
            raise RunError(err)


def add_code_metadata(path=""):
    if path:
        if "://" in path:
            return None
        if os.path.isfile(path):
            path = os.path.dirname(path)
    path = path or "./"

    try:
        from git import (
            GitCommandNotFound,
            InvalidGitRepositoryError,
            NoSuchPathError,
            Repo,
        )
    except ImportError:
        return None

    try:
        repo = Repo(path, search_parent_directories=True)
        remotes = [
            remote.url
            for remote in repo.remotes
            # some stale remotes might be missing urls
            # there is not a nicer way to check for this.
            if repo.config_reader().has_option(f'remote "{remote}"', "url")
        ]
        if len(remotes) > 0:
            return f"{remotes[0]}#{repo.head.commit.hexsha}"

    except (
        InvalidGitRepositoryError,
        NoSuchPathError,
    ):
        # Path is not part of a git repository or an invalid path (will fail later if it needs to)
        pass

    except (GitCommandNotFound, ValueError) as exc:
        logger.warning(
            "Failed to add git metadata",
            path=path,
            error=err_to_str(exc),
        )
    return None


def results_to_iter(results, runspec, execution):
    if not results:
        logger.error("Got an empty results list in to_iter")
        return

    iter = []
    failed = 0
    running = 0
    for task in results:
        if task:
            state = get_in(task, ["status", "state"])
            id = get_in(task, ["metadata", "iteration"])
            struct = {
                "param": get_in(task, ["spec", "parameters"], {}),
                "output": get_in(task, ["status", "results"], {}),
                "state": state,
                "iter": id,
            }
            if state == "error":
                failed += 1
                err = get_in(task, ["status", "error"], "")
                logger.error(f"error in task  {execution.uid}:{id} - {err_to_str(err)}")
            elif state != "completed":
                running += 1

            iter.append(struct)

    if not iter:
        execution.set_state("completed", commit=True)
        logger.warning("Warning!, zero iteration results")
        return
    if hasattr(pd, "json_normalize"):
        df = pd.json_normalize(iter).sort_values("iter")
    else:
        df = pd.io.json.json_normalize(iter).sort_values("iter")
    header = df.columns.values.tolist()
    summary = [header] + df.values.tolist()
    if not runspec:
        return summary, df

    criteria = runspec.spec.hyper_param_options.selector
    item, id = selector(results, criteria)
    if runspec.spec.selector and not id:
        logger.warning(
            f"No best result selected, check selector ({criteria}) or results"
        )
    if id:
        logger.info(f"Best iteration={id}, used criteria {criteria}")
    task = results[item] if id and results else None
    execution.log_iteration_results(id, summary, task)

    log_iter_artifacts(execution, df, header)

    if failed:
        execution.set_state(
            error=f"{failed} of {len(results)} tasks failed, check logs in db for details",
            commit=False,
        )
    elif running == 0:
        execution.set_state("completed", commit=False)
    execution.commit()


def log_iter_artifacts(execution, df, header):
    csv_buffer = StringIO()
    df.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8",
        **mlrun.utils.line_terminator_kwargs(),
    )
    try:
        # may fail due to lack of access credentials to the artifacts store
        execution.log_artifact(
            TableArtifact(
                "iteration_results",
                body=csv_buffer.getvalue(),
                header=header,
                viewer="table",
            ),
            local_path="iteration_results.csv",
        )
        # may also fail due to missing plotly
        execution.log_artifact(
            "parallel_coordinates",
            body=gen_pcp_plot(df, index_col="iter"),
            local_path="parallel_coordinates.html",
        )
    except Exception as exc:
        logger.warning(f"failed to log iter artifacts, {err_to_str(exc)}")


def fill_function_image_name_template(
    registry: str,
    repository: str,
    project: str,
    name: str,
    tag: str,
) -> str:
    image_name_prefix = resolve_function_target_image_name_prefix(project, name)
    return f"{registry}{repository}/{image_name_prefix}:{tag}"


def resolve_function_target_image_name_prefix(project: str, name: str):
    return config.httpdb.builder.function_target_image_name_prefix_template.format(
        project=project, name=name
    )


def resolve_function_target_image_registries_to_enforce_prefix():
    registry, repository = helpers.get_parsed_docker_registry()
    repository = helpers.get_docker_repository_or_default(repository)
    return [
        f"{mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}{repository}/",
        f"{registry}/{repository}/",
    ]


def set_named_item(obj, item):
    if isinstance(item, dict):
        obj[item["name"]] = item
    else:
        obj[item.name] = item


def get_item_name(item, attr="name"):
    if isinstance(item, dict):
        return item.get(attr)
    else:
        return getattr(item, attr, None)


def verify_limits(
    resources_field_name,
    mem=None,
    cpu=None,
    gpus=None,
    gpu_type="nvidia.com/gpu",
):
    if mem:
        verify_field_regex(
            f"function.spec.{resources_field_name}.limits.memory",
            mem,
            mlrun.utils.regex.k8s_resource_quantity_regex
            + mlrun.utils.regex.pipeline_param,
            mode=mlrun.common.schemas.RegexMatchModes.any,
        )
    if cpu:
        verify_field_regex(
            f"function.spec.{resources_field_name}.limits.cpu",
            cpu,
            mlrun.utils.regex.k8s_resource_quantity_regex
            + mlrun.utils.regex.pipeline_param,
            mode=mlrun.common.schemas.RegexMatchModes.any,
        )
    # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
    if gpus:
        verify_field_regex(
            f"function.spec.{resources_field_name}.limits.gpus",
            gpus,
            mlrun.utils.regex.k8s_resource_quantity_regex
            + mlrun.utils.regex.pipeline_param,
            mode=mlrun.common.schemas.RegexMatchModes.any,
        )
    return generate_resources(mem=mem, cpu=cpu, gpus=gpus, gpu_type=gpu_type)


def verify_requests(
    resources_field_name,
    mem=None,
    cpu=None,
):
    if mem:
        verify_field_regex(
            f"function.spec.{resources_field_name}.requests.memory",
            mem,
            mlrun.utils.regex.k8s_resource_quantity_regex
            + mlrun.utils.regex.pipeline_param,
            mode=mlrun.common.schemas.RegexMatchModes.any,
        )
    if cpu:
        verify_field_regex(
            f"function.spec.{resources_field_name}.requests.cpu",
            cpu,
            mlrun.utils.regex.k8s_resource_quantity_regex
            + mlrun.utils.regex.pipeline_param,
            mode=mlrun.common.schemas.RegexMatchModes.any,
        )
    return generate_resources(mem=mem, cpu=cpu)


def get_gpu_from_resource_requirement(requirement: dict):
    """
    Because there could be different types of gpu types, and we don't know all the gpu types possible,
    we want to get the gpu type and its value, we can figure out the type by knowing what resource types are static
    and the possible number of resources.
    Kubernetes support 3 types of resources, two of which their name doesn't change : cpu, memory.
    :param requirement: requirement resource ( limits / requests ) which contain the resources.
    """
    if not requirement:
        return None, None

    if len(requirement) > 3:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Unable to resolve the gpu type because there are more than 3 resources"
        )
    for resource, value in requirement.items():
        if resource not in ["cpu", "memory"]:
            return resource, value
    return None, None


def generate_resources(mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"):
    """get pod cpu/memory/gpu resources dict"""
    resources = {}
    if gpus:
        resources[gpu_type] = gpus
    if mem:
        resources["memory"] = mem
    if cpu:
        resources["cpu"] = cpu
    return resources


def get_func_selector(project, name=None, tag=None):
    s = [f"{mlrun_constants.MLRunInternalLabels.project}={project}"]
    if name:
        s.append(f"{mlrun_constants.MLRunInternalLabels.function}={name}")
        s.append(f"{mlrun_constants.MLRunInternalLabels.tag}={tag or 'latest'}")
    return s


def enrich_function_from_dict(function, function_dict):
    override_function = mlrun.new_function(runtime=function_dict, kind=function.kind)
    for attribute in [
        "volumes",
        "volume_mounts",
        "env",
        "resources",
        "image",
        "image_pull_policy",
        "replicas",
        "node_name",
        "node_selector",
        "affinity",
        "priority_class_name",
        "credentials",
        "tolerations",
        "preemption_mode",
        "security_context",
    ]:
        if attribute == "credentials":
            override_value = getattr(override_function.metadata, attribute, None)
        else:
            override_value = getattr(override_function.spec, attribute, None)
        if override_value:
            if attribute == "env":
                for env_dict in override_value:
                    if env_dict.get("value") is not None:
                        function.set_env(env_dict["name"], env_dict["value"])
                    else:
                        function.set_env(
                            env_dict["name"],
                            value_from=env_dict["valueFrom"],
                        )
            elif attribute == "volumes":
                function.spec.update_vols_and_mounts(override_value, [])
            elif attribute == "volume_mounts":
                # volume mounts don't have a well defined identifier (like name for volume) so we can't merge,
                # only override
                function.spec.volume_mounts = override_value
            elif attribute == "resources":
                # don't override if there are limits and requests but both are empty
                if override_value.get("limits", {}) or override_value.get(
                    "requests", {}
                ):
                    setattr(function.spec, attribute, override_value)
            elif attribute == "credentials":
                if any(override_value.to_dict().values()):
                    function.metadata.credentials = override_value
            else:
                setattr(function.spec, attribute, override_value)
    return function


def enrich_run_labels(
    labels: dict,
    labels_to_enrich: Optional[list[mlrun_constants.MLRunInternalLabels]] = None,
):
    """
    Enrich the run labels with the internal labels and the labels enrichment extension
    :param labels: The run labels dict
    :param labels_to_enrich: The label keys to enrich from MLRunInternalLabels.default_run_labels_to_enrich
    :return: The enriched labels dict
    """
    # Merge the labels with the labels enrichment extension
    labels_enrichment = {
        mlrun_constants.MLRunInternalLabels.owner: os.environ.get("V3IO_USERNAME")
        or getpass.getuser(),
        # TODO: remove this in 1.10.0
        mlrun_constants.MLRunInternalLabels.v3io_user: os.environ.get("V3IO_USERNAME"),
    }

    # Resolve which label keys to enrich
    if labels_to_enrich is None:
        labels_to_enrich = (
            mlrun_constants.MLRunInternalLabels.default_run_labels_to_enrich()
        )

    # Enrich labels
    for label in labels_to_enrich:
        if isinstance(label, enum.Enum):
            label = label.value
        enrichment = labels_enrichment.get(label)
        if label not in labels and enrichment:
            labels[label] = enrichment
    return labels


def resolve_node_selectors(
    project_node_selector: dict, instance_node_selector: dict
) -> dict:
    config_node_selector = mlrun.mlconf.get_default_function_node_selector()
    if project_node_selector or config_node_selector:
        mlrun.utils.logger.debug(
            "Enriching node selector from project and mlrun config",
            project_node_selector=project_node_selector,
            config_node_selector=config_node_selector,
        )
        return mlrun.utils.helpers.merge_dicts_with_precedence(
            config_node_selector,
            project_node_selector,
            instance_node_selector,
        )
    return instance_node_selector


def enrich_gateway_timeout_annotations(annotations: dict, gateway_timeout: int):
    """
    Set gateway proxy connect/read/send timeout annotations
    :param annotations:     The annotations to enrich
    :param gateway_timeout: The timeout to set
    """
    if not gateway_timeout:
        return
    gateway_timeout_str = str(gateway_timeout)
    annotations["nginx.ingress.kubernetes.io/proxy-connect-timeout"] = (
        gateway_timeout_str
    )
    annotations["nginx.ingress.kubernetes.io/proxy-read-timeout"] = gateway_timeout_str
    annotations["nginx.ingress.kubernetes.io/proxy-send-timeout"] = gateway_timeout_str
