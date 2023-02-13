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

import enum
import hashlib
import inspect
import json
import re
import sys
import time
import typing
import warnings
from datetime import datetime, timezone
from importlib import import_module
from os import path
from types import ModuleType
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas
import semver
import yaml
from dateutil import parser
from deprecated import deprecated
from pandas._libs.tslibs.timestamps import Timedelta, Timestamp
from yaml.representer import RepresenterError

import mlrun
import mlrun.errors
import mlrun.utils.version.version
from mlrun.errors import err_to_str

from ..config import config
from .logger import create_logger

yaml.Dumper.ignore_aliases = lambda *args: True
_missing = object()

hub_prefix = "hub://"
DB_SCHEMA = "store"

LEGAL_TIME_UNITS = ["year", "month", "day", "hour", "minute", "second"]
DEFAULT_TIME_PARTITIONS = ["year", "month", "day", "hour"]
DEFAULT_TIME_PARTITIONING_GRANULARITY = "hour"


class StorePrefix:
    """map mlrun store objects to prefixes"""

    FeatureSet = "feature-sets"
    FeatureVector = "feature-vectors"
    Artifact = "artifacts"
    Model = "models"
    Dataset = "datasets"

    @classmethod
    def is_artifact(cls, prefix):
        return prefix in [cls.Artifact, cls.Model, cls.Dataset]

    @classmethod
    def kind_to_prefix(cls, kind):
        kind_map = {"model": cls.Model, "dataset": cls.Dataset}
        return kind_map.get(kind, cls.Artifact)

    @classmethod
    def is_prefix(cls, prefix):
        return prefix in [
            cls.Artifact,
            cls.Model,
            cls.Dataset,
            cls.FeatureSet,
            cls.FeatureVector,
        ]


def get_artifact_target(item: dict, project=None):
    if is_legacy_artifact(item):
        db_key = item.get("db_key")
        project_str = project or item.get("project")
        tree = item.get("tree")
    else:
        db_key = item["spec"].get("db_key")
        project_str = project or item["metadata"].get("project")
        tree = item["metadata"].get("tree")

    kind = item.get("kind")
    if kind in ["dataset", "model"] and db_key:
        return f"{DB_SCHEMA}://{StorePrefix.Artifact}/{project_str}/{db_key}:{tree}"

    return (
        item.get("target_path")
        if is_legacy_artifact(item)
        else item["spec"].get("target_path")
    )


logger = create_logger(config.log_level, config.log_formatter, "mlrun", sys.stdout)
missing = object()

is_ipython = False
try:
    import IPython

    ipy = IPython.get_ipython()
    # if its IPython terminal ignore (cant show html)
    if ipy and "Terminal" not in str(type(ipy)):
        is_ipython = True
except ImportError:
    pass

if is_ipython and config.nest_asyncio_enabled in ["1", "True"]:
    # bypass Jupyter asyncio bug
    import nest_asyncio

    nest_asyncio.apply()


class run_keys:
    input_path = "input_path"
    output_path = "output_path"
    inputs = "inputs"
    artifacts = "artifacts"
    outputs = "outputs"
    data_stores = "data_stores"
    secrets = "secret_sources"


def verify_field_regex(
    field_name, field_value, patterns, raise_on_failure: bool = True
) -> bool:
    for pattern in patterns:
        if not re.match(pattern, str(field_value)):
            log_func = logger.warn if raise_on_failure else logger.debug
            log_func(
                "Field is malformed. Does not match required pattern",
                field_name=field_name,
                field_value=field_value,
                pattern=pattern,
            )
            if raise_on_failure:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Field '{field_name}' is malformed. Does not match required pattern: {pattern}"
                )
            else:
                return False
    return True


def validate_tag_name(
    tag_name: str, field_name: str, raise_on_failure: bool = True
) -> bool:
    """
    This function is used to validate a tag name for invalid characters using field regex.
    if raise_on_failure is set True, throws an MLRunInvalidArgumentError if the tag is invalid,
    otherwise, it returns False
    """
    return mlrun.utils.helpers.verify_field_regex(
        field_name,
        tag_name,
        mlrun.utils.regex.tag_name,
        raise_on_failure=raise_on_failure,
    )


def get_regex_list_as_string(regex_list: List) -> str:
    """
    This function is used to combine a list of regex strings into a single regex,
    with and condition between them.
    """
    return "".join(["(?={regex})".format(regex=regex) for regex in regex_list]) + ".*$"


def tag_name_regex_as_string() -> str:
    return get_regex_list_as_string(mlrun.utils.regex.tag_name)


def is_yaml_path(url):
    return url.endswith(".yaml") or url.endswith(".yml")


# Verifying that a field input is of the expected type. If not the method raises a detailed MLRunInvalidArgumentError
def verify_field_of_type(field_name: str, field_value, expected_type: type):
    if not isinstance(field_value, expected_type):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Field '{field_name}' should be of type {expected_type.__name__} "
            f"(got: {type(field_value).__name__} with value: {field_value})."
        )


# Verifying that a field input is of type list and all elements inside are of the expected element type.
# If not the method raises a detailed MLRunInvalidArgumentError
def verify_field_list_of_type(
    field_name: str, field_value, expected_element_type: type
):
    verify_field_of_type(field_name, field_value, list)
    for element in field_value:
        verify_field_of_type(field_name, element, expected_element_type)


def verify_dict_items_type(
    name: str,
    dictionary: dict,
    expected_keys_types: list = None,
    expected_values_types: list = None,
):
    if dictionary:
        if type(dictionary) != dict:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"{name} expected to be of type dict, got type : {type(dictionary)}"
            )
        try:
            verify_list_items_type(dictionary.keys(), expected_keys_types)
            verify_list_items_type(dictionary.values(), expected_values_types)
        except mlrun.errors.MLRunInvalidArgumentTypeError as exc:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"{name} should be of type Dict[{get_pretty_types_names(expected_keys_types)},"
                f"{get_pretty_types_names(expected_values_types)}]."
            ) from exc


def verify_list_items_type(list_, expected_types: list = None):
    if list_ and expected_types:
        list_items_types = set(map(type, list_))
        expected_types = set(expected_types)

        if not list_items_types.issubset(expected_types):
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"Found unexpected types in list items. expected: {expected_types},"
                f" found: {list_items_types} in : {list_}"
            )


def get_pretty_types_names(types):
    if len(types) == 0:
        return ""
    if len(types) > 1:
        return "Union[" + ",".join([ty.__name__ for ty in types]) + "]"
    return types[0].__name__


def now_date():
    return datetime.now(timezone.utc)


def to_date_str(d):
    if d:
        return d.isoformat()
    return ""


def normalize_name(name):
    # TODO: Must match
    # [a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?
    name = re.sub(r"\s+", "-", name)
    if "_" in name:
        warnings.warn(
            "Names with underscore '_' are about to be deprecated, use dashes '-' instead. "
            "Replacing underscores with dashes.",
            FutureWarning,
        )
        name = name.replace("_", "-")
    return name.lower()


class LogBatchWriter:
    def __init__(self, func, batch=16, maxtime=5):
        self.batch = batch
        self.maxtime = maxtime
        self.start_time = datetime.now()
        self.buffer = ""
        self.func = func

    def write(self, data):
        self.buffer += data
        self.batch -= 1
        elapsed_time = (datetime.now() - self.start_time).seconds
        if elapsed_time > self.maxtime or self.batch <= 0:
            self.flush()

    def flush(self):
        self.func(self.buffer)
        self.buffer = ""
        self.start_time = datetime.now()


def get_in(obj, keys, default=None):
    """
    >>> get_in({'a': {'b': 1}}, 'a.b')
    1
    """
    if isinstance(keys, str):
        keys = keys.split(".")

    for key in keys:
        if not obj or key not in obj:
            return default
        obj = obj[key]
    return obj


def verify_and_update_in(
    obj, key, value, expected_type: type, append=False, replace=True
):
    verify_field_of_type(key, value, expected_type)
    update_in(obj, key, value, append, replace)


def verify_list_and_update_in(
    obj, key, value, expected_element_type: type, append=False, replace=True
):
    verify_field_list_of_type(key, value, expected_element_type)
    update_in(obj, key, value, append, replace)


def _split_by_dots_with_escaping(key: str):
    """
    splits the key by dots, taking escaping into account so that an escaped key can contain dots
    """
    parts = []
    current_key, escape = "", False
    for char in key:
        if char == "." and not escape:
            parts.append(current_key)
            current_key = ""
        elif char == "\\":
            escape = not escape
        else:
            current_key += char
    parts.append(current_key)
    return parts


def update_in(obj, key, value, append=False, replace=True):
    parts = _split_by_dots_with_escaping(key) if isinstance(key, str) else key
    for part in parts[:-1]:
        sub = obj.get(part, missing)
        if sub is missing:
            sub = obj[part] = {}
        obj = sub

    last_key = parts[-1]
    if last_key not in obj:
        if append:
            obj[last_key] = []
        else:
            obj[last_key] = {}

    if append:
        if isinstance(value, list):
            obj[last_key] += value
        else:
            obj[last_key].append(value)
    else:
        if replace or not obj.get(last_key):
            obj[last_key] = value


def match_labels(labels, conditions):
    match = True

    def splitter(verb, text):
        items = text.split(verb)
        if len(items) != 2:
            raise ValueError(f"illegal condition - {text}")
        return labels.get(items[0].strip(), ""), items[1].strip()

    for condition in conditions:
        if "~=" in condition:
            l, val = splitter("~=", condition)
            match = match and val in l
        elif "!=" in condition:
            l, val = splitter("!=", condition)
            match = match and val != l
        elif "=" in condition:
            l, val = splitter("=", condition)
            match = match and val == l
        else:
            match = match and (condition.strip() in labels)
    return match


def match_times(time_from, time_to, obj, key):
    obj_time = get_in(obj, key)
    if not obj_time:
        # if obj doesn't have the required time, return false if either time_from or time_to were given
        return not time_from and not time_to
    obj_time = parser.isoparse(obj_time)

    if (time_from and time_from > obj_time) or (time_to and time_to < obj_time):
        return False

    return True


def match_value(value, obj, key):
    if not value:
        return True
    return get_in(obj, key, _missing) == value


def match_value_options(value_options, obj, key):
    if not value_options:
        return True

    return get_in(obj, key, _missing) in as_list(value_options)


def flatten(df, col, prefix=""):
    params = []
    for r in df[col]:
        if r:
            for k in r.keys():
                if k not in params:
                    params += [k]
    for p in params:
        df[prefix + p] = df[col].apply(lambda x: x.get(p, "") if x else "")
    df.drop(col, axis=1, inplace=True)
    return df


def list2dict(lines: list):
    out = {}
    for line in lines:
        i = line.find("=")
        if i == -1:
            continue
        key, value = line[:i].strip(), line[i + 1 :].strip()
        if key is None:
            raise ValueError("cannot find key in line (key=value)")
        value = path.expandvars(value)
        out[key] = value
    return out


def dict_to_list(struct: dict):
    if not struct:
        return []
    return [f"{k}={v}" for k, v in struct.items()]


def dict_to_str(struct: dict, sep=","):
    return sep.join(dict_to_list(struct))


def numpy_representer_seq(dumper, data):
    return dumper.represent_list(data.tolist())


def float_representer(dumper, data):
    return dumper.represent_float(data)


def int_representer(dumper, data):
    return dumper.represent_int(data)


def date_representer(dumper, data):
    if isinstance(data, np.datetime64):
        value = str(data)
    else:
        value = data.isoformat()
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", value)


def enum_representer(dumper, data):
    return dumper.represent_str(str(data.value))


yaml.add_representer(np.int64, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.integer, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.floating, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.ndarray, numpy_representer_seq, Dumper=yaml.SafeDumper)
yaml.add_representer(np.datetime64, date_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(Timestamp, date_representer, Dumper=yaml.SafeDumper)
yaml.add_multi_representer(enum.Enum, enum_representer, Dumper=yaml.SafeDumper)


def dict_to_yaml(struct) -> str:
    try:
        data = yaml.safe_dump(struct, default_flow_style=False, sort_keys=False)
    except RepresenterError as exc:
        raise ValueError("error: data result cannot be serialized to YAML") from exc
    return data


# solve numpy json serialization
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, str, float, list, dict)):
            return obj
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)


def dict_to_json(struct):
    return json.dumps(struct, cls=MyEncoder)


def parse_versioned_object_uri(uri, default_project=""):
    project = default_project
    tag = ""
    hash_key = ""
    if "/" in uri:
        loc = uri.find("/")
        project = uri[:loc]
        uri = uri[loc + 1 :]
    if ":" in uri:
        loc = uri.find(":")
        tag = uri[loc + 1 :]
        uri = uri[:loc]
    if "@" in uri:
        loc = uri.find("@")
        hash_key = uri[loc + 1 :]
        uri = uri[:loc]

    return project, uri, tag, hash_key


def parse_artifact_uri(uri, default_project=""):
    uri_pattern = r"^((?P<project>.*)/)?(?P<key>.*?)(\#(?P<iteration>.*?))?(:(?P<tag>.*?))?(@(?P<uid>.*))?$"
    match = re.match(uri_pattern, uri)
    if not match:
        raise ValueError(
            "Uri not in supported format [<project>/]<key>[#<iteration>][:<tag>][@<uid>]"
        )
    group_dict = match.groupdict()
    iteration = group_dict["iteration"]
    if iteration is not None:
        try:
            iteration = int(iteration)
        except ValueError:
            raise ValueError(
                f"illegal store path {uri}, iteration must be integer value"
            )
    return (
        group_dict["project"] or default_project,
        group_dict["key"],
        iteration,
        group_dict["tag"],
        group_dict["uid"],
    )


def generate_object_uri(project, name, tag=None, hash_key=None):
    uri = f"{project}/{name}"

    # prioritize hash key over tag
    if hash_key:
        uri += f"@{hash_key}"
    elif tag:
        uri += f":{tag}"
    return uri


def generate_artifact_uri(project, key, tag=None, iter=None):
    artifact_uri = f"{project}/{key}"
    if iter is not None:
        artifact_uri = f"{artifact_uri}#{iter}"
    if tag is not None:
        artifact_uri = f"{artifact_uri}:{tag}"
    return artifact_uri


def extend_hub_uri_if_needed(uri):
    if not uri.startswith(hub_prefix):
        return uri, False
    name = uri[len(hub_prefix) :]
    tag = "master"
    if ":" in name:
        loc = name.find(":")
        tag = name[loc + 1 :]
        name = name[:loc]

    # hub function directory name are with underscores instead of hyphens
    name = name.replace("-", "_")
    return config.get_hub_url().format(name=name, tag=tag), True


def gen_md_table(header, rows=None):
    rows = [] if rows is None else rows

    def gen_list(items=None):
        items = [] if items is None else items
        out = "|"
        for i in items:
            out += f" {i} |"
        return out

    out = gen_list(header) + "\n" + gen_list(len(header) * ["---"]) + "\n"
    for r in rows:
        out += gen_list(r) + "\n"
    return out


def gen_html_table(header, rows=None):
    rows = [] if rows is None else rows

    style = """
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-style:solid;border-width:1px;padding:6px 4px;}
.tg th{font-weight:normal;border-style:solid;border-width:1px;padding:6px 4px;}
</style>
"""

    def gen_list(items=None, tag="td"):
        items = [] if items is None else items
        out = ""
        for item in items:
            out += f"<{tag}>{item}</{tag}>"
        return out

    out = "<tr>" + gen_list(header, "th") + "</tr>\n"
    for r in rows:
        out += "<tr>" + gen_list(r, "td") + "</tr>\n"
    return style + '<table class="tg">\n' + out + "</table>\n\n"


def new_pipe_metadata(
    artifact_path: str = None,
    cleanup_ttl: int = None,
    op_transformers: typing.List[typing.Callable] = None,
):
    from kfp.dsl import PipelineConf

    def _set_artifact_path(task):
        from kubernetes import client as k8s_client

        task.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_ARTIFACT_PATH", value=artifact_path)
        )
        return task

    conf = PipelineConf()
    cleanup_ttl = cleanup_ttl or int(config.kfp_ttl)

    if cleanup_ttl:
        conf.set_ttl_seconds_after_finished(cleanup_ttl)
    if artifact_path:
        conf.add_op_transformer(_set_artifact_path)
    if op_transformers:
        for op_transformer in op_transformers:
            conf.add_op_transformer(op_transformer)
    return conf


# TODO: remove in 1.5.0
@deprecated(
    version="1.3.0",
    reason="'new_pipe_meta' will be removed in 1.5.0",
    category=FutureWarning,
)
def new_pipe_meta(artifact_path=None, ttl=None, *args):
    return new_pipe_metadata(
        artifact_path=artifact_path, cleanup_ttl=ttl, op_transformers=args
    )


def _convert_python_package_version_to_image_tag(version: typing.Optional[str]):
    return (
        version.replace("+", "-").replace("0.0.0-", "") if version is not None else None
    )


def enrich_image_url(
    image_url: str, client_version: str = None, client_python_version: str = None
) -> str:
    client_version = _convert_python_package_version_to_image_tag(client_version)
    server_version = _convert_python_package_version_to_image_tag(
        mlrun.utils.version.Version().get()["version"]
    )
    image_url = image_url.strip()
    mlrun_version = config.images_tag or client_version or server_version
    tag = mlrun_version
    tag += resolve_image_tag_suffix(
        mlrun_version=mlrun_version, python_version=client_python_version
    )
    registry = config.images_registry

    # it's an mlrun image if the repository is mlrun
    is_mlrun_image = image_url.startswith("mlrun/") or "/mlrun/" in image_url

    if is_mlrun_image and tag and ":" not in image_url:
        image_url = f"{image_url}:{tag}"

    enrich_registry = False
    # enrich registry only if images_to_enrich_registry provided
    # example: "^mlrun/*" means enrich only if the image repository is mlrun and registry is not specified (in which
    # case /mlrun/ will be part of the url)

    if config.images_to_enrich_registry:
        for pattern_to_enrich in config.images_to_enrich_registry.split(","):
            if re.match(pattern_to_enrich, image_url):
                enrich_registry = True
    if registry and enrich_registry:
        registry = registry if registry.endswith("/") else f"{registry}/"
        image_url = f"{registry}{image_url}"

    return image_url


def resolve_image_tag_suffix(
    mlrun_version: str = None, python_version: str = None
) -> str:
    """
    resolves what suffix should be appended to the image tag
    :param mlrun_version: the mlrun version
    :param python_version: the requested python version
    :return: the suffix to append to the image tag
    """
    if not python_version or not mlrun_version:
        return ""

    # if the mlrun version is 0.0.0-<unstable>/<commit hash> then it's a dev version, therefore we can't check if the
    # mlrun version is higher than 1.3.0, but we can check the python version and if python version was passed it
    # means it 1.3.0-rc or higher, so we can add the suffix of the python version.
    if mlrun_version.startswith("0.0.0-") or "unstable" in mlrun_version:
        if python_version.startswith("3.7"):
            return "-py37"
        return ""

    # For mlrun 1.3.0, we decided to support mlrun runtimes images with both python 3.7 and 3.9 images.
    # While the python 3.9 images will continue to have no suffix, the python 3.7 images will have a '-py37' suffix.
    # Python 3.8 images will not be supported for mlrun 1.3.0, meaning that if the user has client with python 3.8
    # and mlrun 1.3.x then the image will be pulled without a suffix (which is the python 3.9 image).
    # using semver (x.y.z-X) to include rc versions as well
    if semver.VersionInfo.parse(mlrun_version) >= semver.VersionInfo.parse(
        "1.3.0-X"
    ) and python_version.startswith("3.7"):
        return "-py37"
    return ""


def get_docker_repository_or_default(repository: str) -> str:
    if not repository:
        repository = "mlrun"
    return repository


def get_parsed_docker_registry() -> Tuple[Optional[str], Optional[str]]:
    # according to https://stackoverflow.com/questions/37861791/how-are-docker-image-names-parsed
    docker_registry = config.httpdb.builder.docker_registry
    first_slash_index = docker_registry.find("/")
    # this is exception to the rules from the link above, since the config value is called docker_registry we assume
    # that if someone gave just one component without any slash they gave a registry and not a repository
    if first_slash_index == -1:
        return docker_registry, None
    if (
        docker_registry[:first_slash_index].find(".") == -1
        and docker_registry[:first_slash_index].find(":") == -1
        and docker_registry[:first_slash_index] != "localhost"
    ):
        return None, docker_registry
    else:
        return (
            docker_registry[:first_slash_index],
            docker_registry[first_slash_index + 1 :],
        )


def fill_object_hash(object_dict, uid_property_name, tag=""):
    # remove tag, hash, date from calculation
    object_dict.setdefault("metadata", {})
    tag = tag or object_dict["metadata"].get("tag")
    status = object_dict.setdefault("status", {})
    object_dict["metadata"]["tag"] = ""
    object_dict["metadata"][uid_property_name] = ""
    object_dict["status"] = None
    object_dict["metadata"]["updated"] = None
    object_created_timestamp = object_dict["metadata"].pop("created", None)
    data = json.dumps(object_dict, sort_keys=True).encode()
    h = hashlib.sha1()
    h.update(data)
    uid = h.hexdigest()
    object_dict["metadata"]["tag"] = tag
    object_dict["metadata"][uid_property_name] = uid
    object_dict["status"] = status
    if object_created_timestamp:
        object_dict["metadata"]["created"] = object_created_timestamp
    return uid


def fill_function_hash(function_dict, tag=""):
    return fill_object_hash(function_dict, "hash", tag)


def create_linear_backoff(base=2, coefficient=2, stop_value=120):
    """
    Create a generator of linear backoff. Check out usage example in test_helpers.py
    """
    x = 0
    comparison = min if coefficient >= 0 else max

    while True:
        next_value = comparison(base + x * coefficient, stop_value)
        yield next_value
        x += 1


def create_step_backoff(steps=None):
    """
    Create a generator of steps backoff.
    Example: steps = [[2, 5], [20, 10], [120, None]] will produce a generator in which the first 5
    values will be 2, the next 10 values will be 20 and the rest will be 120.
    :param steps: a list of lists [step_value, number_of_iteration_in_this_step]
    """
    steps = steps if steps is not None else [[2, 10], [10, 10], [120, None]]
    steps = iter(steps)

    # Get first step
    step = next(steps)
    while True:
        current_step_value, current_step_remain = step
        if current_step_remain == 0:

            # No more in this step, moving on
            step = next(steps)
        elif current_step_remain is None:

            # We are in the last step, staying here forever
            yield current_step_value
        elif current_step_remain > 0:

            # Still more remains in this step, just reduce the remaining number
            step[1] -= 1
            yield current_step_value


def create_exponential_backoff(base=2, max_value=120, scale_factor=1):
    """
    Create a generator of exponential backoff. Check out usage example in test_helpers.py
    :param base: exponent base
    :param max_value: max limit on the result
    :param scale_factor: factor to be used as linear scaling coefficient
    """
    exponent = 1
    while True:

        # This "complex" implementation (unlike the one in linear backoff) is to avoid exponent growing too fast and
        # risking going behind max_int
        next_value = scale_factor * (base**exponent)
        if next_value < max_value:
            exponent += 1
            yield next_value
        else:
            yield max_value


def retry_until_successful(
    backoff: int, timeout: int, logger, verbose: bool, _function, *args, **kwargs
):
    """
    Runs function with given *args and **kwargs.
    Tries to run it until success or timeout reached (timeout is optional)
    :param backoff: can either be a:
            - number (int / float) that will be used as interval.
            - generator of waiting intervals. (support next())
    :param timeout: pass None if timeout is not wanted, number of seconds if it is
    :param logger: a logger so we can log the failures
    :param verbose: whether to log the failure on each retry
    :param _function: function to run
    :param args: functions args
    :param kwargs: functions kwargs
    :return: function result
    """
    start_time = time.time()
    last_exception = None

    # Check if backoff is just a simple interval
    if isinstance(backoff, int) or isinstance(backoff, float):
        backoff = create_linear_backoff(base=backoff, coefficient=0)

    first_interval = next(backoff)
    if timeout and timeout <= first_interval:
        logger.warning(
            f"timeout ({timeout}) must be higher than backoff ({first_interval})."
            f" Set timeout to be higher than backoff."
        )

    # If deadline was not provided or deadline not reached
    while timeout is None or time.time() < start_time + timeout:
        next_interval = first_interval or next(backoff)
        first_interval = None
        try:
            result = _function(*args, **kwargs)
            return result

        except mlrun.errors.MLRunFatalFailureError as exc:
            raise exc.original_exception
        except Exception as exc:
            last_exception = exc

            # If next interval is within allowed time period - wait on interval, abort otherwise
            if timeout is None or time.time() + next_interval < start_time + timeout:
                if logger is not None and verbose:
                    logger.debug(
                        f"Operation not yet successful, Retrying in {next_interval} seconds."
                        f" exc: {err_to_str(exc)}"
                    )

                time.sleep(next_interval)
            else:
                break

    if logger is not None:
        logger.warning(
            f"Operation did not complete on time. last exception: {last_exception}"
        )

    raise Exception(
        f"failed to execute command by the given deadline."
        f" last_exception: {last_exception},"
        f" function_name: {_function.__name__},"
        f" timeout: {timeout}"
    )


def get_ui_url(project, uid=None):
    url = ""
    if mlrun.mlconf.resolve_ui_url():
        url = "{}/{}/{}/jobs".format(
            mlrun.mlconf.resolve_ui_url(), mlrun.mlconf.ui.projects_prefix, project
        )
        if uid:
            url += f"/monitor/{uid}/overview"
    return url


def get_workflow_url(project, id=None):
    url = ""
    if mlrun.mlconf.resolve_ui_url():
        url = "{}/{}/{}/jobs/monitor-workflows/workflow/{}".format(
            mlrun.mlconf.resolve_ui_url(), mlrun.mlconf.ui.projects_prefix, project, id
        )
    return url


def are_strings_in_exception_chain_messages(
    exception: Exception, strings_list=typing.List[str]
) -> bool:
    while exception is not None:
        if any([string in str(exception) for string in strings_list]):
            return True
        exception = exception.__cause__
    return False


def create_class(pkg_class: str):
    """Create a class from a package.module.class string

    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    splits = pkg_class.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    class_ = getattr(import_module(".".join(pkg_module)), clfclass)
    return class_


def create_function(pkg_func: list):
    """Create a function from a package.module.function string

    :param pkg_func:  full function location,
                      e.g. "sklearn.feature_selection.f_classif"
    """
    splits = pkg_func.split(".")
    pkg_module = ".".join(splits[:-1])
    cb_fname = splits[-1]
    pkg_module = __import__(pkg_module, fromlist=[cb_fname])
    function_ = getattr(pkg_module, cb_fname)
    return function_


def get_caller_globals(level=2):
    try:
        return inspect.stack()[level][0].f_globals
    except Exception:
        return None


def _module_to_namespace(namespace):
    if isinstance(namespace, ModuleType):
        members = inspect.getmembers(
            namespace, lambda o: inspect.isfunction(o) or isinstance(o, type)
        )
        return {key: mod for key, mod in members}
    return namespace


def _search_in_namespaces(name, namespaces):
    """search the class/function in a list of modules"""
    if not namespaces:
        return None
    if not isinstance(namespaces, list):
        namespaces = [namespaces]
    for namespace in namespaces:
        namespace = _module_to_namespace(namespace)
        if name in namespace:
            return namespace[name]
    return None


def get_class(class_name, namespace=None):
    """return class object from class name string"""
    if isinstance(class_name, type):
        return class_name
    class_object = _search_in_namespaces(class_name, namespace)
    if class_object is not None:
        return class_object

    try:
        class_object = create_class(class_name)
    except (ImportError, ValueError) as exc:
        raise ImportError(f"Failed to import {class_name}") from exc
    return class_object


def get_function(function, namespace):
    """return function callable object from function name string"""
    if callable(function):
        return function

    function = function.strip()
    if function.startswith("("):
        if not function.endswith(")"):
            raise ValueError('function expression must start with "(" and end with ")"')
        return eval("lambda event: " + function[1:-1], {}, {})
    function_object = _search_in_namespaces(function, namespace)
    if function_object is not None:
        return function_object

    try:
        function_object = create_function(function)
    except (ImportError, ValueError) as exc:
        raise ImportError(
            f"state/function init failed, handler {function} not found"
        ) from exc
    return function_object


def get_handler_extended(
    handler_path: str, context=None, class_args: dict = {}, namespaces=None
):
    """get function handler from [class_name::]handler string

    :param handler_path:  path to the function ([class_name::]handler)
    :param context:       MLRun function/job client context
    :param class_args:    optional dict of class init kwargs
    :param namespaces:    one or list of namespaces/modules to search the handler in
    :return: function handler (callable)
    """
    if "::" not in handler_path:
        return get_function(handler_path, namespaces)

    splitted = handler_path.split("::")
    class_path = splitted[0].strip()
    handler_path = splitted[1].strip()

    class_object = get_class(class_path, namespaces)
    argspec = inspect.getfullargspec(class_object)
    if argspec.varkw or "context" in argspec.args:
        class_args["context"] = context
    try:
        instance = class_object(**class_args)
    except TypeError as exc:
        raise TypeError(
            f"failed to init class {class_path}\n args={class_args}"
        ) from exc

    if not hasattr(instance, handler_path):
        raise ValueError(
            f"handler ({handler_path}) specified but doesnt exist in class {class_path}"
        )
    return getattr(instance, handler_path)


def datetime_from_iso(time_str: str) -> Optional[datetime]:
    if not time_str:
        return
    return parser.isoparse(time_str)


def datetime_to_iso(time_obj: Optional[datetime]) -> Optional[str]:
    if not time_obj:
        return
    return time_obj.isoformat()


def as_list(element: Any) -> List[Any]:
    return element if isinstance(element, list) else [element]


def calculate_local_file_hash(filename):
    h = hashlib.sha1()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def calculate_dataframe_hash(dataframe: pandas.DataFrame):
    # https://stackoverflow.com/questions/49883236/how-to-generate-a-hash-or-checksum-value-on-python-dataframe-created-from-a-fix/62754084#62754084
    return hashlib.sha1(pandas.util.hash_pandas_object(dataframe).values).hexdigest()


def fill_artifact_path_template(artifact_path, project):
    # Supporting {{project}} is new, in certain setup configuration the default artifact path has the old
    # {{run.project}} so we're supporting it too for backwards compatibility
    if artifact_path and (
        "{{run.project}}" in artifact_path or "{{project}}" in artifact_path
    ):
        if not project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "project name must be specified with this"
                + f" artifact_path template {artifact_path}"
            )
        artifact_path = artifact_path.replace("{{run.project}}", project)
        artifact_path = artifact_path.replace("{{project}}", project)
    return artifact_path


def str_to_timestamp(time_str: str, now_time: Timestamp = None):
    """convert fixed/relative time string to Pandas Timestamp

    can use relative times using the "now" verb, and align to floor using the "floor" verb

    time string examples::

        1/1/2021
        now
        now + 1d2h
        now -1d floor 1H
    """
    if not isinstance(time_str, str):
        return time_str

    time_str = time_str.strip()
    if time_str.lower().startswith("now"):
        # handle now +/- timedelta
        timestamp: Timestamp = now_time or Timestamp.now()
        time_str = time_str[len("now") :].lstrip()
        split = time_str.split("floor")
        time_str = split[0].strip()

        if time_str and time_str[0] in ["+", "-"]:
            timestamp = timestamp + Timedelta(time_str)
        elif time_str:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"illegal time string expression now{time_str}, "
                'use "now +/- <timestring>" for relative times'
            )

        if len(split) > 1:
            timestamp = timestamp.floor(split[1].strip())
        return timestamp

    return Timestamp(time_str)


def is_legacy_artifact(artifact):
    if isinstance(artifact, dict):
        return "metadata" not in artifact
    else:
        return not hasattr(artifact, "metadata")


def get_in_artifact(artifact: dict, key, default=None, raise_on_missing=False):
    """artifact can be dict or Artifact object"""
    if is_legacy_artifact(artifact):
        return artifact.get(key, default)
    elif key == "kind":
        return artifact.get(key, default)
    else:
        for block in ["metadata", "spec", "status"]:
            block_obj = artifact.get(block, {})
            if block_obj and key in block_obj:
                return block_obj.get(key, default)

        if raise_on_missing:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"artifact {artifact} is missing metadata/spec/status"
            )
        return default


def set_paths(pythonpath=""):
    """update the sys path"""
    if not pythonpath:
        return
    paths = pythonpath.split(":")
    for p in paths:
        abspath = path.abspath(p)
        if abspath not in sys.path:
            sys.path.append(abspath)


def is_relative_path(path):
    if not path:
        return False
    return not (path.startswith("/") or ":\\" in path or "://" in path)


def as_number(field_name, field_value):
    if isinstance(field_value, str) and not field_value.isnumeric():
        raise ValueError(f"{field_name} must be numeric (str/int types)")
    return int(field_value)


def filter_warnings(action, category):
    def decorator(function):
        def wrapper(*args, **kwargs):

            # context manager that copies and, upon exit, restores the warnings filter and the showwarning() function.
            with warnings.catch_warnings():
                warnings.simplefilter(action, category)
                return function(*args, **kwargs)

        return wrapper

    return decorator
