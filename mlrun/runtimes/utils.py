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
import hashlib
import json
import os
import re
import typing
from copy import deepcopy
from io import StringIO
from sys import stderr

import pandas as pd
from kubernetes import client

import mlrun
import mlrun.builder
import mlrun.utils.regex
from mlrun.api.utils.clients import nuclio
from mlrun.db import get_run_db
from mlrun.errors import err_to_str
from mlrun.frameworks.parallel_coordinates import gen_pcp_plot
from mlrun.k8s_utils import get_k8s_helper
from mlrun.runtimes.constants import MPIJobCRDVersions

from ..artifacts import TableArtifact
from ..config import config
from ..utils import get_in, helpers, logger, verify_field_regex
from .generators import selector


class RunError(Exception):
    pass


mlrun_key = "mlrun/"


class _ContextStore:
    def __init__(self):
        self._context = None

    def get(self):
        return self._context

    def set(self, context):
        self._context = context


global_context = _ContextStore()


cached_mpijob_crd_version = None
cached_nuclio_version = None


# resolve mpijob runtime according to the mpi-operator's supported crd-version
# if specified on mlrun config set it likewise,
# if not specified, try resolving it according to the mpi-operator, otherwise set to default
# since this is a heavy operation (sending requests to k8s/API), and it's unlikely that the crd version
# will change in any context - cache it
def resolve_mpijob_crd_version(api_context=False):
    global cached_mpijob_crd_version
    if not cached_mpijob_crd_version:

        # config override everything
        mpijob_crd_version = config.mpijob_crd_version

        if not mpijob_crd_version:
            in_k8s_cluster = get_k8s_helper(
                silent=True
            ).is_running_inside_kubernetes_cluster()
            if in_k8s_cluster:
                k8s_helper = get_k8s_helper()
                namespace = k8s_helper.resolve_namespace()

                # try resolving according to mpi-operator that's running
                res = k8s_helper.list_pods(
                    namespace=namespace, selector="component=mpi-operator"
                )
                if len(res) > 0:
                    mpi_operator_pod = res[0]
                    mpijob_crd_version = mpi_operator_pod.metadata.labels.get(
                        "crd-version"
                    )
            elif not in_k8s_cluster and not api_context:
                # connect will populate the config from the server config
                # TODO: something nicer
                get_run_db()
                mpijob_crd_version = config.mpijob_crd_version

            # If resolution failed simply use default
            if not mpijob_crd_version:
                mpijob_crd_version = MPIJobCRDVersions.default()

        if mpijob_crd_version not in MPIJobCRDVersions.all():
            raise ValueError(
                f"unsupported mpijob crd version: {mpijob_crd_version}. "
                f"supported versions: {MPIJobCRDVersions.all()}"
            )
        cached_mpijob_crd_version = mpijob_crd_version

    return cached_mpijob_crd_version


def resolve_spark_operator_version():
    try:
        regex = re.compile("spark-([23])")
        return int(regex.findall(config.spark_operator_version)[0])
    except Exception:
        raise ValueError("Failed to resolve spark operator's version")


# if nuclio version specified on mlrun config set it likewise,
# if not specified, get it from nuclio api client
# since this is a heavy operation (sending requests to API), and it's unlikely that the version
# will change - cache it (this means if we upgrade nuclio, we need to restart mlrun to re-fetch the new version)
def resolve_nuclio_version():
    global cached_nuclio_version

    if not cached_nuclio_version:

        # config override everything
        nuclio_version = config.nuclio_version
        if not nuclio_version and config.nuclio_dashboard_url:
            try:
                nuclio_client = nuclio.Client()
                nuclio_version = nuclio_client.get_dashboard_version()
            except Exception as exc:
                logger.warning("Failed to resolve nuclio version", exc=err_to_str(exc))

        cached_nuclio_version = nuclio_version

    return cached_nuclio_version


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
        logger.error(f"exec error - {err_to_str(err)}")
        print(err, file=stderr)
        if not silent:
            raise RunError(err)


class AsyncLogWriter:
    def __init__(self, db, runobj):
        self.db = db
        self.uid = runobj.metadata.uid
        self.project = runobj.metadata.project or ""
        self.iter = runobj.metadata.iteration

    def write(self, data):
        if self.db:
            self.db.store_log(self.uid, self.project, data, append=True)

    def flush(self):
        # todo: verify writes are large enough, if not cache and use flush
        pass


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
        remotes = [remote.url for remote in repo.remotes]
        if len(remotes) > 0:
            return f"{remotes[0]}#{repo.head.commit.hexsha}"
    except (GitCommandNotFound, InvalidGitRepositoryError, NoSuchPathError, ValueError):
        pass
    return None


def set_if_none(struct, key, value):
    if not struct.get(key):
        struct[key] = value


def results_to_iter(results, runspec, execution):
    if not results:
        logger.error("got an empty results list in to_iter")
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
        logger.warning("warning!, zero iteration results")
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
            f"no best result selected, check selector ({criteria}) or results"
        )
    if id:
        logger.info(f"best iteration={id}, used criteria {criteria}")
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
    df.to_csv(csv_buffer, index=False, line_terminator="\n", encoding="utf-8")
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


def resolve_function_image_name(function, image: typing.Optional[str] = None) -> str:
    project = function.metadata.project or config.default_project
    name = function.metadata.name
    tag = function.metadata.tag or "latest"
    if image:
        image_name_prefix = resolve_function_target_image_name_prefix(project, name)
        registries_to_enforce_prefix = (
            resolve_function_target_image_registries_to_enforce_prefix()
        )
        for registry in registries_to_enforce_prefix:
            if image.startswith(registry):
                prefix_with_registry = f"{registry}{image_name_prefix}"
                if not image.startswith(prefix_with_registry):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Configured registry enforces image name to start with this prefix: {image_name_prefix}"
                    )
        return image
    return generate_function_image_name(project, name, tag)


def generate_function_image_name(project: str, name: str, tag: str) -> str:
    _, repository = helpers.get_parsed_docker_registry()
    repository = helpers.get_docker_repository_or_default(repository)
    return fill_function_image_name_template(
        mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX, repository, project, name, tag
    )


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
        f"{mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}{repository}/",
        f"{registry}/{repository}/",
    ]


def set_named_item(obj, item):
    if isinstance(item, dict):
        obj[item["name"]] = item
    else:
        obj[item.name] = item


def set_item_attribute(item, attribute, value):
    if isinstance(item, dict):
        item[attribute] = value
    else:
        setattr(item, attribute, value)


def get_item_name(item, attr="name"):
    if isinstance(item, dict):
        return item.get(attr)
    else:
        return getattr(item, attr, None)


def apply_kfp(modify, cop, runtime):
    modify(cop)

    # Have to do it here to avoid circular dependencies
    from .pod import AutoMountType

    if AutoMountType.is_auto_modifier(modify):
        runtime.spec.disable_auto_mount = True

    api = client.ApiClient()
    for k, v in cop.pod_labels.items():
        runtime.metadata.labels[k] = v
    for k, v in cop.pod_annotations.items():
        runtime.metadata.annotations[k] = v
    if cop.container.env:
        env_names = [
            e.name if hasattr(e, "name") else e["name"] for e in runtime.spec.env
        ]
        for e in api.sanitize_for_serialization(cop.container.env):
            name = e["name"]
            if name in env_names:
                runtime.spec.env[env_names.index(name)] = e
            else:
                runtime.spec.env.append(e)
                env_names.append(name)
        cop.container.env.clear()

    if cop.volumes and cop.container.volume_mounts:
        vols = api.sanitize_for_serialization(cop.volumes)
        mounts = api.sanitize_for_serialization(cop.container.volume_mounts)
        runtime.spec.update_vols_and_mounts(vols, mounts)
        cop.volumes.clear()
        cop.container.volume_mounts.clear()

    return runtime


def get_resource_labels(function, run=None, scrape_metrics=None):
    scrape_metrics = (
        scrape_metrics if scrape_metrics is not None else config.scrape_metrics
    )
    run_uid, run_name, run_project, run_owner = None, None, None, None
    if run:
        run_uid = run.metadata.uid
        run_name = run.metadata.name
        run_project = run.metadata.project
        run_owner = run.metadata.labels.get("owner")
    labels = deepcopy(function.metadata.labels)
    labels[mlrun_key + "class"] = function.kind
    labels[mlrun_key + "project"] = run_project or function.metadata.project
    labels[mlrun_key + "function"] = str(function.metadata.name)
    labels[mlrun_key + "tag"] = str(function.metadata.tag or "latest")
    labels[mlrun_key + "scrape-metrics"] = str(scrape_metrics)

    if run_uid:
        labels[mlrun_key + "uid"] = run_uid

    if run_name:
        labels[mlrun_key + "name"] = run_name

    if run_owner:
        labels[mlrun_key + "owner"] = run_owner

    return labels


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
            mlrun.utils.regex.k8s_resource_quantity_regex,
        )
    if cpu:
        verify_field_regex(
            f"function.spec.{resources_field_name}.limits.cpu",
            cpu,
            mlrun.utils.regex.k8s_resource_quantity_regex,
        )
    # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
    if gpus:
        verify_field_regex(
            f"function.spec.{resources_field_name}.limits.gpus",
            gpus,
            mlrun.utils.regex.k8s_resource_quantity_regex,
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
            mlrun.utils.regex.k8s_resource_quantity_regex,
        )
    if cpu:
        verify_field_regex(
            f"function.spec.{resources_field_name}.requests.cpu",
            cpu,
            mlrun.utils.regex.k8s_resource_quantity_regex,
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
    s = [f"{mlrun_key}project={project}"]
    if name:
        s.append(f"{mlrun_key}function={name}")
        s.append(f"{mlrun_key}tag={tag or 'latest'}")
    return s


def parse_function_selector(selector: typing.List[str]) -> typing.Tuple[str, str, str]:
    project, name, tag = None, None, None
    for criteria in selector:
        if f"{mlrun_key}project=" in criteria:
            project = criteria[f"{mlrun_key}project=":]
        if f"{mlrun_key}function=" in criteria:
            name = criteria[f"{mlrun_key}function=":]
        if f"{mlrun_key}tag=" in criteria:
            tag = criteria[f"{mlrun_key}tag=":]
    return project, name, tag


class k8s_resource:
    kind = ""
    per_run = False
    per_function = False
    k8client = None

    def deploy_function(self, function):
        pass

    def release_function(self, function):
        pass

    def submit_run(self, function, runobj):
        pass

    def get_object(self, name, namespace=None):
        return None

    def get_status(self, name, namespace=None):
        return None

    def del_object(self, name, namespace=None):
        pass

    def get_pods(self, name, namespace=None, master=False):
        return {}


def enrich_function_from_dict(function, function_dict):
    override_function = mlrun.new_function(runtime=function_dict, kind=function.kind)
    for attribute in [
        "volumes",
        "volume_mounts",
        "env",
        "resources",
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
