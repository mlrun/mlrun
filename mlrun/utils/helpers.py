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

import asyncio
import base64
import enum
import functools
import gzip
import hashlib
import inspect
import itertools
import json
import os
import re
import string
import sys
import traceback
import typing
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from importlib import import_module, reload
from os import path
from types import ModuleType
from typing import Any, Optional
from urllib.parse import urlparse

import git
import inflection
import numpy as np
import packaging.version
import pandas
import pytz
import semver
import yaml
from dateutil import parser
from pandas import Timedelta, Timestamp
from yaml.representer import RepresenterError

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.helpers
import mlrun.common.runtimes.constants as runtimes_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.regex
import mlrun.utils.version.version
import mlrun_pipelines.common.constants
import mlrun_pipelines.models
import mlrun_pipelines.utils
from mlrun.common.constants import MYSQL_MEDIUMBLOB_SIZE_BYTES
from mlrun.config import config
from mlrun_pipelines.models import PipelineRun

from .logger import create_logger
from .retryer import (  # noqa: F401
    AsyncRetryer,
    Retryer,
    create_exponential_backoff,
    create_linear_backoff,
    create_step_backoff,
)

yaml.Dumper.ignore_aliases = lambda *args: True
_missing = object()

hub_prefix = "hub://"
DB_SCHEMA = "store"

LEGAL_TIME_UNITS = ["year", "month", "day", "hour", "minute", "second"]
DEFAULT_TIME_PARTITIONS = ["year", "month", "day", "hour"]
DEFAULT_TIME_PARTITIONING_GRANULARITY = "hour"


class OverwriteBuildParamsWarning(FutureWarning):
    pass


class StorePrefix:
    """map mlrun store objects to prefixes"""

    FeatureSet = "feature-sets"
    FeatureVector = "feature-vectors"
    Artifact = "artifacts"
    Model = "models"
    Dataset = "datasets"
    Document = "documents"

    @classmethod
    def is_artifact(cls, prefix):
        return prefix in [cls.Artifact, cls.Model, cls.Dataset, cls.Document]

    @classmethod
    def kind_to_prefix(cls, kind):
        kind_map = {
            "model": cls.Model,
            "dataset": cls.Dataset,
            "document": cls.Document,
        }
        return kind_map.get(kind, cls.Artifact)

    @classmethod
    def is_prefix(cls, prefix):
        return prefix in [
            cls.Artifact,
            cls.Model,
            cls.Dataset,
            cls.FeatureSet,
            cls.FeatureVector,
            cls.Document,
        ]


def get_artifact_target(item: dict, project=None):
    db_key = item["spec"].get("db_key")
    project_str = project or item["metadata"].get("project")
    tree = item["metadata"].get("tree")
    tag = item["metadata"].get("tag")
    iter = item["metadata"].get("iter")
    kind = item.get("kind")
    uid = item["metadata"].get("uid")

    if kind in {"dataset", "model", "artifact"} and db_key:
        target = (
            f"{DB_SCHEMA}://{StorePrefix.kind_to_prefix(kind)}/{project_str}/{db_key}"
        )
        if iter:
            target = f"{target}#{iter}"
        target += f":{tag}" if tag else ":latest"
        if tree:
            target += f"@{tree}"
        if uid:
            target += f"^{uid}"
        return target

    return item["spec"].get("target_path")


# TODO: Remove once data migration v5 is obsolete
def is_legacy_artifact(artifact):
    if isinstance(artifact, dict):
        return "metadata" not in artifact
    else:
        return not hasattr(artifact, "metadata")


logger = create_logger(config.log_level, config.log_formatter, "mlrun", sys.stdout)
missing = object()

is_ipython = False  # is IPython terminal, including Jupyter
is_jupyter = False  # is Jupyter notebook/lab terminal
try:
    import IPython.core.getipython

    ipy = IPython.core.getipython.get_ipython()

    is_ipython = ipy is not None
    is_jupyter = (
        is_ipython
        # not IPython
        and "Terminal" not in str(type(ipy))
    )

    del ipy
except ModuleNotFoundError:
    pass

if is_jupyter and config.nest_asyncio_enabled in ["1", "True"]:
    # bypass Jupyter asyncio bug
    import nest_asyncio

    nest_asyncio.apply()


class RunKeys:
    input_path = "input_path"
    output_path = "output_path"
    inputs = "inputs"
    returns = "returns"
    artifacts = "artifacts"
    artifact_uris = "artifact_uris"
    outputs = "outputs"
    data_stores = "data_stores"
    secrets = "secret_sources"


# for Backward compatibility
run_keys = RunKeys


def verify_field_regex(
    field_name,
    field_value,
    patterns,
    raise_on_failure: bool = True,
    log_message: str = "Field is malformed. Does not match required pattern",
    mode: mlrun.common.schemas.RegexMatchModes = mlrun.common.schemas.RegexMatchModes.all,
) -> bool:
    # limit the error message
    max_chars = 63
    for pattern in patterns:
        if not re.match(pattern, str(field_value)):
            log_func = logger.warn if raise_on_failure else logger.debug
            log_func(
                log_message,
                field_name=field_name,
                field_value=field_value,
                pattern=pattern,
            )
            if mode == mlrun.common.schemas.RegexMatchModes.all:
                if raise_on_failure:
                    if len(field_name) > max_chars:
                        field_name = field_name[:max_chars] + "...truncated"
                    if len(field_value) > max_chars:
                        field_value = field_value[:max_chars] + "...truncated"
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Field '{field_name}' is malformed. '{field_value}' "
                        f"does not match required pattern: {pattern}"
                    )
                return False
        elif mode == mlrun.common.schemas.RegexMatchModes.any:
            return True
    if mode == mlrun.common.schemas.RegexMatchModes.all:
        return True
    elif mode == mlrun.common.schemas.RegexMatchModes.any:
        if raise_on_failure:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Field '{field_name[:max_chars]}' is malformed. '{field_value[:max_chars]}' does not match any of the"
                f" required patterns: {patterns}"
            )
        return False


def validate_builder_source(
    source: str, pull_at_runtime: bool = False, workdir: Optional[str] = None
):
    if pull_at_runtime or not source:
        return

    if "://" not in source:
        if not path.isabs(source):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Source '{source}' must be a valid URL or absolute path when 'pull_at_runtime' is False "
                "set 'source' to a remote URL to clone/copy the source to the base image, "
                "or set 'pull_at_runtime' to True to pull the source at runtime."
            )

        else:
            logger.warn(
                "Loading local source at build time requires the source to be on the base image, "
                "in which case it is recommended to use 'workdir' instead",
                source=source,
                workdir=workdir,
            )

    if source.endswith(".zip"):
        logger.warn(
            "zip files are not natively extracted by docker, use tar.gz for faster loading during build",
            source=source,
        )


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
        log_message="Special characters are not permitted in tag names",
    )


def validate_artifact_key_name(
    artifact_key: str, field_name: str, raise_on_failure: bool = True
) -> bool:
    field_type = "key" if field_name == "artifact.key" else "db_key"
    return mlrun.utils.helpers.verify_field_regex(
        field_name,
        artifact_key,
        mlrun.utils.regex.artifact_key,
        raise_on_failure=raise_on_failure,
        log_message=f"Artifact {field_type} must start and end with an alphanumeric character, and may only contain "
        "letters, numbers, hyphens, underscores, and dots.",
    )


def validate_inline_artifact_body_size(body: typing.Union[str, bytes, None]) -> None:
    if body and len(body) > MYSQL_MEDIUMBLOB_SIZE_BYTES:
        raise mlrun.errors.MLRunBadRequestError(
            "The body of the artifact exceeds the maximum allowed size. "
            "Avoid embedding the artifact body. "
            "This increases the size of the project yaml file and could affect the project during loading and saving. "
            "More information is available at"
            "https://docs.mlrun.org/en/latest/projects/automate-project-git-source.html#setting-and-registering-the-project-artifacts"
        )


def validate_v3io_stream_consumer_group(
    value: str, raise_on_failure: bool = True
) -> bool:
    return mlrun.utils.helpers.verify_field_regex(
        "consumerGroup",
        value,
        mlrun.utils.regex.v3io_stream_consumer_group,
        raise_on_failure=raise_on_failure,
    )


def get_regex_list_as_string(regex_list: list) -> str:
    """
    This function is used to combine a list of regex strings into a single regex,
    with and condition between them.
    """
    return "".join([f"(?={regex})" for regex in regex_list]) + ".*$"


def tag_name_regex_as_string() -> str:
    return get_regex_list_as_string(mlrun.utils.regex.tag_name)


def is_yaml_path(url):
    return url.endswith(".yaml") or url.endswith(".yml")


def remove_image_protocol_prefix(image: str) -> str:
    if not image:
        return image

    prefixes = ["https://", "https://"]
    if any(prefix in image for prefix in prefixes):
        image = image.removeprefix("https://").removeprefix("http://")
        logger.warning(
            "The image has an unexpected protocol prefix ('http://' or 'https://'). "
            "If you wish to use the default configured registry, no protocol prefix is required "
            "(note that you can also use '.<image-name>' instead of the full URL where <image-name> is a placeholder). "
            "Removing protocol prefix from image.",
            image=image,
        )
    return image


# Verifying that a field input is of the expected type. If not the method raises a detailed MLRunInvalidArgumentError
def verify_field_of_type(field_name: str, field_value, expected_type: type):
    if not isinstance(field_value, expected_type):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Field '{field_name}' should be of type '{expected_type.__name__}' "
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
    expected_keys_types: Optional[list] = None,
    expected_values_types: Optional[list] = None,
):
    if dictionary:
        if not isinstance(dictionary, dict):
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"'{name}' expected to be of type dict, got type: {type(dictionary)}"
            )
        try:
            verify_list_items_type(dictionary.keys(), expected_keys_types)
            verify_list_items_type(dictionary.values(), expected_values_types)
        except mlrun.errors.MLRunInvalidArgumentTypeError as exc:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                f"'{name}' should be of type Dict[{get_pretty_types_names(expected_keys_types)}, "
                f"{get_pretty_types_names(expected_values_types)}]."
            ) from exc


def verify_list_items_type(list_, expected_types: Optional[list] = None):
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


def now_date(tz: timezone = timezone.utc) -> datetime:
    return datetime.now(tz=tz)


def datetime_to_mysql_ts(datetime_object: datetime) -> datetime:
    """
    Convert a Python datetime object to a MySQL-compatible timestamp string,
    rounded to the nearest millisecond.
    Example: 2024-12-18T16:36:05.235687+00:00 -> 2024-12-18T16:36:05.236000

    :param datetime_object: A Python datetime object.

    :return: A MySQL-compatible timestamp string with millisecond precision.
    """
    if not datetime_object.tzinfo:
        datetime_object = datetime_object.replace(tzinfo=timezone.utc)

    # Round to the nearest millisecond
    ms = round(datetime_object.microsecond / 1000) * 1000
    if ms == 1000000:
        datetime_object += timedelta(seconds=1)
        ms = 0

    return datetime_object.replace(microsecond=ms)


def datetime_min(tz: timezone = timezone.utc) -> datetime:
    return datetime(1970, 1, 1, tzinfo=tz)


datetime_now = now_date


def to_date_str(d):
    if d:
        return d.isoformat()
    return ""


def normalize_name(name: str, verbose: bool = True):
    # TODO: Must match
    # [a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?
    name = re.sub(r"\s+", "-", name)
    if "_" in name:
        if verbose:
            warnings.warn(
                "Names with underscore '_' are about to be deprecated, use dashes '-' instead. "
                f"Replacing '{name}' underscores with dashes.",
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
    >>> get_in({"a": {"b": 1}}, "a.b")
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
            left, val = splitter("~=", condition)
            match = match and val in left
        elif "!=" in condition:
            left, val = splitter("!=", condition)
            match = match and val != left
        elif "=" in condition:
            left, val = splitter("=", condition)
            match = match and val == left
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


def parse_artifact_uri(uri, default_project=""):
    """
    Parse artifact URI into project, key, tag, iter, tree, uid
    URI format: [<project>/]<key>[#<iter>][:<tag>][@<tree>][^<uid>]

    :param uri:            uri to parse
    :param default_project: default project name if not in URI
    :returns: a tuple of:
        [0] = project name
        [1] = key
        [2] = iteration
        [3] = tag
        [4] = tree
        [5] = uid
    """
    uri_pattern = mlrun.utils.regex.artifact_uri_pattern
    match = re.match(uri_pattern, uri)
    if not match:
        raise ValueError(
            "Uri not in supported format [<project>/]<key>[#<iteration>][:<tag>][@<tree>]"
        )
    group_dict = match.groupdict()
    iteration = group_dict["iteration"]
    if iteration is not None:
        try:
            iteration = int(iteration)
        except ValueError:
            raise ValueError(
                f"illegal store path '{uri}', iteration must be integer value"
            )
    else:
        iteration = 0
    return (
        group_dict["project"] or default_project,
        group_dict["key"],
        iteration,
        group_dict["tag"],
        group_dict["tree"],
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


def generate_artifact_uri(
    project, key, tag=None, iter=None, tree=None, uid=None
) -> str:
    artifact_uri = f"{project}/{key}"
    if iter is not None:
        artifact_uri = f"{artifact_uri}#{iter}"
    if tag is not None:
        artifact_uri = f"{artifact_uri}:{tag}"
    if tree is not None:
        artifact_uri = f"{artifact_uri}@{tree}"
    if uid is not None:
        artifact_uri = f"{artifact_uri}^{uid}"
    return artifact_uri


def extend_hub_uri_if_needed(uri) -> tuple[str, bool]:
    """
    Retrieve the full uri of the item's yaml in the hub.

    :param uri: structure: "hub://[<source>/]<item-name>[:<tag>]"

    :return: A tuple of:
               [0] = Extended URI of item
               [1] =  Is hub item (bool)
    """
    is_hub_uri = uri.startswith(hub_prefix)
    if not is_hub_uri:
        return uri, is_hub_uri

    db = mlrun.get_run_db()
    name = uri.removeprefix(hub_prefix)
    tag = "latest"
    source_name = ""
    if ":" in name:
        name, tag = name.split(":")
    if "/" in name:
        try:
            source_name, name = name.split("/")
        except ValueError as exc:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid character '/' in function name or source name"
            ) from exc
    name = normalize_name(name=name, verbose=False)
    if not source_name:
        # Searching item in all sources
        sources = db.list_hub_sources(item_name=name, tag=tag)
        if not sources:
            raise mlrun.errors.MLRunNotFoundError(
                f"Item={name}, tag={tag} not found in any hub source"
            )
        # precedence to user source
        indexed_source = sources[0]
    else:
        # Specific source is given
        indexed_source = db.get_hub_source(source_name)
    # hub function directory name are with underscores instead of hyphens
    name = name.replace("-", "_")
    function_suffix = f"{name}/{tag}/src/function.yaml"
    return indexed_source.source.get_full_uri(function_suffix), is_hub_uri


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


def _convert_python_package_version_to_image_tag(version: typing.Optional[str]):
    return (
        version.replace("+", "-").replace("0.0.0-", "") if version is not None else None
    )


def enrich_image_url(
    image_url: str,
    client_version: Optional[str] = None,
    client_python_version: Optional[str] = None,
) -> str:
    image_url = image_url.strip()

    # Add python version tag if needed
    if image_url == "python" and client_python_version:
        image_tag = ".".join(client_python_version.split(".")[:2])
        image_url = f"python:{image_tag}"

    client_version = _convert_python_package_version_to_image_tag(client_version)
    server_version = _convert_python_package_version_to_image_tag(
        mlrun.utils.version.Version().get()["version"]
    )
    mlrun_version = config.images_tag or client_version or server_version
    tag = mlrun_version or ""

    # TODO: Remove condition when mlrun/mlrun-kfp image is also supported
    if "mlrun-kfp" not in image_url:
        tag += resolve_image_tag_suffix(
            mlrun_version=mlrun_version, python_version=client_python_version
        )

    # it's an mlrun image if the repository is mlrun
    is_mlrun_image = image_url.startswith("mlrun/") or "/mlrun/" in image_url

    if is_mlrun_image and tag and ":" not in image_url:
        image_url = f"{image_url}:{tag}"

    registry = (
        config.images_registry if is_mlrun_image else config.vendor_images_registry
    )

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
    mlrun_version: Optional[str] = None, python_version: Optional[str] = None
) -> str:
    """
    Resolves what suffix to be appended to the image tag
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
        if python_version.startswith("3.9"):
            return "-py39"
        return ""

    # For mlrun 1.9.x and 1.10.x, we support mlrun runtimes images with both python 3.9 and 3.11 images.
    # While the python 3.11 images will continue to have no suffix, the python 3.9 images will have a '-py39' suffix.
    # Python 3.10 images are not supported in mlrun 1.9.0, meaning that if the user has client with python 3.10
    # and mlrun 1.9.x then the image will be pulled without a suffix (which is the python 3.11 image).
    # using semver (x.y.z-X) to include rc versions as well
    if semver.VersionInfo.parse("1.11.0-X") > semver.VersionInfo.parse(
        mlrun_version
    ) >= semver.VersionInfo.parse("1.9.0-X") and python_version.startswith("3.9"):
        return "-py39"
    return ""


def get_docker_repository_or_default(repository: str) -> str:
    if not repository:
        repository = "mlrun"
    return repository


def get_parsed_docker_registry() -> tuple[Optional[str], Optional[str]]:
    # according to https://stackoverflow.com/questions/37861791/how-are-docker-image-names-parsed
    docker_registry = config.httpdb.builder.docker_registry or ""
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

    # Note the usage of default=str here, which means everything not JSON serializable (for example datetime) will be
    # converted to string when dumping to JSON. This is not safe for de-serializing (since it won't know we
    # originated from a datetime, for example), but since this is a one-way dump only for hash calculation,
    # it's valid here.
    data = json.dumps(object_dict, sort_keys=True, default=str).encode()
    h = hashlib.sha1()
    h.update(data)
    uid = h.hexdigest()

    # restore original values
    object_dict["metadata"]["tag"] = tag
    object_dict["metadata"][uid_property_name] = uid
    object_dict["status"] = status
    if object_created_timestamp:
        object_dict["metadata"]["created"] = object_created_timestamp
    return uid


def fill_function_hash(function_dict, tag=""):
    return fill_object_hash(function_dict, "hash", tag)


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
    return Retryer(backoff, timeout, logger, verbose, _function, *args, **kwargs).run()


async def retry_until_successful_async(
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
    return await AsyncRetryer(
        backoff, timeout, logger, verbose, _function, *args, **kwargs
    ).run()


def get_project_url(project: str) -> str:
    """
    Generate the base URL for a given project.

    :param project: The project name.
    :return: The base URL for the project, or an empty string if the base URL is not resolved.
    """
    if mlrun.mlconf.resolve_ui_url():
        return f"{mlrun.mlconf.resolve_ui_url()}/{mlrun.mlconf.ui.projects_prefix}/{project}"
    return ""


def get_run_url(project: str, uid: str, name: str) -> str:
    """
    Generate the URL for a specific run.

    :param project: The project name.
    :param uid: The run UID.
    :param name: The run name.
    :return: The URL for the run, or an empty string if the base URL is not resolved.
    """
    runs_url = get_runs_url(project)
    if not runs_url:
        return ""
    return f"{runs_url}/monitor-jobs/{name}/{uid}/overview"


def get_runs_url(project: str) -> str:
    """
    Generate the URL for the runs of a given project.

    :param project: The project name.
    :return: The URL for the runs, or an empty string if the base URL is not resolved.
    """
    base_url = get_project_url(project)
    if not base_url:
        return ""
    return f"{base_url}/jobs"


def get_model_endpoint_url(
    project: str,
    model_name: Optional[str] = None,
    model_endpoint_id: Optional[str] = None,
) -> str:
    """
    Generate the URL for a specific model endpoint.

    :param project: The project name.
    :param model_name: The model name.
    :param model_endpoint_id: The model endpoint ID.
    :return: The URL for the model endpoint, or an empty string if the base URL is not resolved.
    """
    base_url = get_project_url(project)
    if not base_url:
        return ""
    url = f"{base_url}/models"
    if model_name and model_endpoint_id:
        url += f"/model-endpoints/{model_name}/{model_endpoint_id}/overview"
    return url


def get_workflow_url(
    project: str,
    id: Optional[str] = None,
) -> str:
    """
    Generate the URL for a specific workflow.

    :param project: The project name.
    :param id: The workflow ID.
    :return: The URL for the workflow, or an empty string if the base URL is not resolved.
    """
    base_url = get_project_url(project)
    if not base_url:
        return ""
    url = f"{base_url}/jobs/monitor-workflows/workflow"
    if id:
        url += f"/{id}"
    return url


def get_kfp_list_runs_filter(
    project_name: Optional[str] = None,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> str:
    """
    Generates a filter for listing Kubeflow Pipelines (KFP) runs.

    :param project_name: The name of the project. If "*", it won't filter by project.
    :param end_date: The latest creation date for filtering runs (ISO 8601 format).
    :param start_date: The earliest creation date for filtering runs (ISO 8601 format).
    :return: A JSON-formatted filter string for KFP.
    """

    # KFP filter operation codes
    kfp_less_than_or_equal_op = 7  # '<='
    kfp_greater_than_or_equal_op = 5  # '>='
    kfp_substring_op = 9  # Substring match

    filters = {"predicates": []}

    if end_date:
        filters["predicates"].append(
            {
                "key": "created_at",
                "op": kfp_less_than_or_equal_op,
                "timestamp_value": end_date,
            }
        )

    if project_name and project_name != "*":
        filters["predicates"].append(
            {
                "key": "name",
                "op": kfp_substring_op,
                "string_value": project_name,
            }
        )
    if start_date:
        filters["predicates"].append(
            {
                "key": "created_at",
                "op": kfp_greater_than_or_equal_op,
                "timestamp_value": start_date,
            }
        )
    return json.dumps(filters)


def validate_and_convert_date(date_input: str) -> str:
    """
    Converts any recognizable date string into a standardized RFC 3339 format.
    :param date_input: A date string in a recognizable format.
    """
    try:
        dt_object = parser.parse(date_input)
        if dt_object.tzinfo is not None:
            # Convert to UTC if it's in a different timezone
            dt_object = dt_object.astimezone(pytz.utc)
        else:
            # If no timezone info is present, assume it's in local time
            local_tz = pytz.timezone("UTC")
            dt_object = local_tz.localize(dt_object)

        # Convert the datetime object to an RFC 3339-compliant string.
        # RFC 3339 requires timestamps to be in ISO 8601 format with a 'Z' suffix for UTC time.
        # The isoformat() method adds a "+00:00" suffix for UTC by default,
        # so we replace it with "Z" to ensure compliance.
        formatted_date = dt_object.isoformat().replace("+00:00", "Z")
        formatted_date = formatted_date.rstrip("Z") + "Z"

        return formatted_date
    except (ValueError, OverflowError) as e:
        raise ValueError(
            f"Invalid date format: {date_input}."
            f" Date format must adhere to the RFC 3339 standard (e.g., 'YYYY-MM-DDTHH:MM:SSZ' for UTC)."
        ) from e


def are_strings_in_exception_chain_messages(
    exception: Exception, strings_list: list[str]
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


def create_function(pkg_func: str, reload_modules: bool = False):
    """Create a function from a package.module.function string

    :param pkg_func:  full function location,
                      e.g. "sklearn.feature_selection.f_classif"
    :param reload_modules: reload the function again.
    """
    splits = pkg_func.split(".")
    pkg_module = ".".join(splits[:-1])
    cb_fname = splits[-1]
    pkg_module = __import__(pkg_module, fromlist=[cb_fname])

    if reload_modules:
        # Even though the function appears in the modules list, we need to reload
        # the code again because it may have changed
        try:
            logger.debug("Reloading module", module=pkg_func)
            _reload(
                pkg_module,
                max_recursion_depth=mlrun.mlconf.function.spec.reload_max_recursion_depth,
            )
        except Exception as exc:
            logger.warning(
                "Failed to reload module. Not all associated modules can be reloaded, import them manually."
                "Or, with Jupyter, restart the Python kernel.",
                module=pkg_func,
                err=mlrun.errors.err_to_str(exc),
            )

    function_ = getattr(pkg_module, cb_fname)
    return function_


def get_caller_globals():
    """Returns a dictionary containing the first non-mlrun caller function's namespace."""
    try:
        stack = inspect.stack()
        # If an API function called this function directly, the first non-mlrun caller will be 2 levels up the stack.
        # Otherwise, we keep going up the stack until we find it.
        for level in range(2, len(stack)):
            namespace = stack[level][0].f_globals
            if (not namespace["__name__"].startswith("mlrun.")) and (
                not namespace["__name__"].startswith("deprecated.")
            ):
                return namespace
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


def get_function(function, namespaces, reload_modules: bool = False):
    """Return function callable object from function name string

    :param function: path to the function ([class_name::]function)
    :param namespaces: one or list of namespaces/modules to search the function in
    :param reload_modules: reload the function again
    :return: function handler (callable)
    """
    if callable(function):
        return function

    function = function.strip()
    if function.startswith("("):
        if not function.endswith(")"):
            raise ValueError('function expression must start with "(" and end with ")"')
        return eval("lambda event: " + function[1:-1], {}, {})
    function_object = _search_in_namespaces(function, namespaces)
    if function_object is not None:
        return function_object

    try:
        function_object = create_function(function, reload_modules)
    except (ImportError, ValueError) as exc:
        raise ImportError(
            f"state/function init failed, handler '{function}' not found"
        ) from exc
    return function_object


def get_handler_extended(
    handler_path: str,
    context=None,
    class_args: Optional[dict] = None,
    namespaces=None,
    reload_modules: bool = False,
):
    """Get function handler from [class_name::]handler string

    :param handler_path:  path to the function ([class_name::]handler)
    :param context:       MLRun function/job client context
    :param class_args:    optional dict of class init kwargs
    :param namespaces:    one or list of namespaces/modules to search the handler in
    :param reload_modules: reload the function again
    :return: function handler (callable)
    """
    class_args = class_args or {}
    if "::" not in handler_path:
        return get_function(handler_path, namespaces, reload_modules)

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
    dt = parser.isoparse(time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # ensure the datetime is in UTC, converting if necessary
    return dt.astimezone(timezone.utc)


def datetime_to_iso(time_obj: Optional[datetime]) -> Optional[str]:
    if not time_obj:
        return
    return time_obj.isoformat()


def enrich_datetime_with_tz_info(timestamp_string) -> Optional[datetime]:
    if not timestamp_string:
        return timestamp_string

    if timestamp_string and not mlrun.utils.helpers.has_timezone(timestamp_string):
        timestamp_string += datetime.now(timezone.utc).astimezone().strftime("%z")

    for _format in [
        # e.g: 2021-08-25 12:00:00.000Z
        "%Y-%m-%d %H:%M:%S.%f%z",
        # e.g: 2024-11-11 07:44:56+0000
        "%Y-%m-%d %H:%M:%S%z",
    ]:
        try:
            return datetime.strptime(timestamp_string, _format)
        except ValueError as exc:
            last_exc = exc
    raise last_exc


def has_timezone(timestamp):
    try:
        dt = parser.parse(timestamp) if isinstance(timestamp, str) else timestamp

        # Check if the parsed datetime object has timezone information
        return dt.tzinfo is not None
    except ValueError:
        return False


def format_datetime(dt: datetime, fmt: Optional[str] = None) -> str:
    if dt is None:
        return ""

    # If the datetime is naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # TODO: Once Python 3.12 is the minimal version, use %:z to format the timezone offset with a colon
    formatted_time = dt.strftime(fmt or "%Y-%m-%d %H:%M:%S.%f%z")

    # For versions earlier than Python 3.12, we manually insert the colon in the timezone offset
    return formatted_time[:-2] + ":" + formatted_time[-2:]


def as_list(element: Any) -> list[Any]:
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


def template_artifact_path(artifact_path, project, run_uid=None):
    """
    Replace {{run.uid}} with the run uid and {{project}} with the project name in the artifact path.
    If no run uid is provided, the word `project` will be used instead as it is assumed to be a project
    level artifact.
    """
    if not artifact_path:
        return artifact_path
    run_uid = run_uid or "project"
    artifact_path = artifact_path.replace("{{run.uid}}", run_uid)
    artifact_path = _fill_project_path_template(artifact_path, project)
    return artifact_path


def _fill_project_path_template(artifact_path, project):
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


def to_non_empty_values_dict(input_dict: dict) -> dict:
    return {key: value for key, value in input_dict.items() if value}


def get_enriched_gpu_limits(function_limits: dict) -> dict[str, int]:
    """
    Creates new limits containing the GPU-related limits from the function's limits,
    mapping each to zero. This is used for pods like Kaniko and Argo pods, which inherit
    GPU-related selectors but do not require GPU resources. By setting these
    limits to zero, the pods receive the necessary tolerations from the cloud provider for scheduling,
    without actually consuming GPU resources.
    """
    return {resource: 0 for resource in function_limits if "/gpu" in resource.lower()}


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


def str_to_bool(value: str) -> bool:
    """Convert a string to a boolean value."""
    value = value.lower()
    if value in ("true", "1", "t", "y", "yes", "on"):
        return True
    if value in ("false", "0", "f", "n", "no", "off"):
        return False
    raise ValueError(f"invalid boolean value: {value}")


def is_link_artifact(artifact):
    if isinstance(artifact, dict):
        return (
            artifact.get("kind") == mlrun.common.schemas.ArtifactCategories.link.value
        )
    else:
        return artifact.kind == mlrun.common.schemas.ArtifactCategories.link.value


def format_run(run: PipelineRun, with_project=False) -> dict:
    fields = [
        "id",
        "name",
        "status",
        "error",
        "created_at",
        "scheduled_at",
        "finished_at",
        "description",
        "experiment_id",
    ]

    if with_project:
        fields.append("project")

    # create a run object that contains all fields,
    run = {
        key: str(value) if value is not None else value
        for key, value in run.items()
        if key in fields
    }

    # if the time_keys values is from 1970, this indicates that the field has not yet been specified yet,
    # and we want to return a None value instead
    time_keys = ["scheduled_at", "finished_at", "created_at"]

    for key, value in run.items():
        if (
            key in time_keys
            and isinstance(value, (str, datetime))
            and parser.parse(str(value)).year == 1970
        ):
            run[key] = None

    # pipelines are yet to populate the status or workflow has failed
    # as observed https://jira.iguazeng.com/browse/ML-5195
    # set to unknown to ensure a status is returned
    if run.get("status", None) is None:
        run["status"] = inflection.titleize(
            mlrun.common.runtimes.constants.RunStates.unknown
        )

    return run


def get_in_artifact(artifact: dict, key, default=None, raise_on_missing=False):
    """artifact can be dict or Artifact object"""
    if key == "kind":
        return artifact.get(key, default)
    else:
        for block in ["metadata", "spec", "status"]:
            block_obj = artifact.get(block, {})
            if block_obj and key in block_obj:
                return block_obj.get(key, default)

        if raise_on_missing:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"artifact '{artifact}' is missing metadata/spec/status"
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


def is_running_in_jupyter_notebook() -> bool:
    """
    Check if the code is running inside a Jupyter Notebook.
    :return: True if running inside a Jupyter Notebook, False otherwise.
    """
    return is_jupyter


def create_ipython_display():
    """
    Create an IPython display object and fill it with initial content.
    We can later use the returned display_id with the update_display method to update the content.
    If IPython is not installed, a warning will be logged and None will be returned.
    """
    if is_ipython:
        import IPython

        display_id = uuid.uuid4().hex
        content = IPython.display.HTML(
            f'<div id="{display_id}">Temporary Display Content</div>'
        )
        IPython.display.display(content, display_id=display_id)
        return display_id

    # returning None if IPython is not installed, this method shouldn't be called in that case but logging for sanity
    logger.debug("IPython is not installed, cannot create IPython display")


def as_number(field_name, field_value):
    if isinstance(field_value, str) and not field_value.isnumeric():
        raise ValueError(f"'{field_name}' must be numeric (str/int types)")
    return int(field_value)


def filter_warnings(action, category):
    """
    Decorator to filter warnings

    Example::
        @filter_warnings("ignore", FutureWarning)
        def my_function():
            pass

    :param action:      one of "error", "ignore", "always", "default", "module", or "once"
    :param category:    a class that the warning must be a subclass of
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            # context manager that copies and, upon exit, restores the warnings filter and the showwarning() function.
            with warnings.catch_warnings():
                warnings.simplefilter(action, category)
                return function(*args, **kwargs)

        return wrapper

    return decorator


def resolve_git_reference_from_source(source):
    # kaniko allow multiple "#" e.g. #refs/..#commit
    split_source = source.split("#", 1)

    # no reference was passed
    if len(split_source) < 2:
        return source, "", ""

    reference = split_source[1]
    if reference.startswith("refs/"):
        return split_source[0], reference, ""

    return split_source[0], "", reference


def ensure_git_branch(url: str, repo: git.Repo) -> str:
    """Ensures git url includes branch.
    If no branch or refs are included in the git source then will enrich the git url with the current active branch
     as defined in the repo object. Otherwise, will just return the url and won't apply any enrichments.

    :param url:   Git source url
    :param repo: `git.Repo` object that will be used for getting the active branch value (if required)

    :return:     Git source url with full valid path to the relevant branch

    """
    source, reference, branch = resolve_git_reference_from_source(url)
    if not branch and not reference:
        url = f"{url}#refs/heads/{repo.active_branch}"
    return url


def is_file_path(filepath):
    root, ext = os.path.splitext(filepath)
    return os.path.isfile(filepath) and ext


def normalize_workflow_name(name, project_name):
    return name.removeprefix(project_name + "-")


def normalize_project_username(username: str):
    username = username.lower()

    # remove domain if exists
    username = username.split("@")[0]

    # replace non r'a-z0-9\-_' chars with empty string
    username = inflection.parameterize(username, separator="")

    # replace underscore with dashes
    username = inflection.dasherize(username)

    # ensure ends with alphanumeric
    username = username.rstrip("-_")

    return username


async def run_in_threadpool(func, *args, **kwargs):
    """
    Run a sync-function in the loop default thread pool executor pool and await its result.
    Note that this function is not suitable for CPU-bound tasks, as it will block the event loop.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)


def is_explicit_ack_supported(context):
    # list from https://github.com/nuclio/nuclio/blob/1.12.0/pkg/platform/abstract/platform.go#L1546
    return hasattr(context, "trigger") and context.trigger in [
        "v3io-stream",
        "v3ioStream",
        "kafka-cluster",
        "kafka",
    ]


def line_terminator_kwargs():
    # pandas 1.5.0 renames line_terminator to lineterminator
    line_terminator_parameter = (
        "lineterminator"
        if packaging.version.Version(pandas.__version__)
        >= packaging.version.Version("1.5.0")
        else "line_terminator"
    )
    return {line_terminator_parameter: "\n"}


def iterate_list_by_chunks(
    iterable_list: typing.Iterable, chunk_size: int
) -> typing.Iterable:
    """
    Iterate over a list and yield chunks of the list in the given chunk size
    e.g.: for list of [a,b,c,d,e,f] and chunk_size of 2, will yield [a,b], [c,d], [e,f]
    """
    if chunk_size <= 0 or not iterable_list:
        yield iterable_list
        return
    iterator = iter(iterable_list)
    while chunk := list(itertools.islice(iterator, chunk_size)):
        yield chunk


def to_parquet(df, *args, **kwargs):
    import pyarrow.lib

    # version set for pyspark compatibility, and is needed as of pyarrow 13 due to timestamp incompatibility
    if "version" not in kwargs:
        kwargs["version"] = "2.4"
    try:
        df.to_parquet(*args, **kwargs)
    except pyarrow.lib.ArrowInvalid as ex:
        if re.match(
            "Fragment would be written into [0-9]+. partitions. This exceeds the maximum of [0-9]+",
            str(ex),
        ):
            raise mlrun.errors.MLRunRuntimeError(
                """Maximum number of partitions exceeded. To resolve this, change
partition granularity by setting time_partitioning_granularity or partition_cols, or disable partitioning altogether by
setting partitioned=False"""
            ) from ex
        else:
            raise ex


def is_ecr_url(registry: str) -> bool:
    # example URL: <aws_account_id>.dkr.ecr.<region>.amazonaws.com
    parsed_url = urlparse(f"https://{registry}")
    hostname = parsed_url.hostname
    return hostname and ".ecr." in hostname and hostname.endswith(".amazonaws.com")


def get_local_file_schema() -> list:
    # The expression `list(string.ascii_lowercase)` generates a list of lowercase alphabets,
    # which corresponds to drive letters in Windows file paths such as `C:/Windows/path`.
    return ["file"] + list(string.ascii_lowercase)


def is_safe_path(base, filepath, is_symlink=False):
    # Avoid path traversal attacks by ensuring that the path is safe
    resolved_filepath = (
        os.path.abspath(filepath) if not is_symlink else os.path.realpath(filepath)
    )
    return base == os.path.commonpath((base, resolved_filepath))


def get_serving_spec():
    data = None

    # we will have the serving spec in either mounted config map
    # or env depending on the size of the spec and configuration

    try:
        with open(mlrun.common.constants.MLRUN_SERVING_SPEC_PATH) as f:
            data = f.read()
    except FileNotFoundError:
        pass

    if data is None:
        data = os.environ.get("SERVING_SPEC_ENV", "")
        if not data:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Failed to find serving spec in env var or config file"
            )
    # Attempt to decode and decompress, or use as-is for backward compatibility
    try:
        decoded_data = base64.b64decode(data)
        decompressed_data = gzip.decompress(decoded_data)
        spec = json.loads(decompressed_data.decode("utf-8"))
    except (OSError, gzip.BadGzipFile, base64.binascii.Error, json.JSONDecodeError):
        spec = json.loads(data)

    return spec


def additional_filters_warning(additional_filters, class_name):
    if additional_filters and any(additional_filters):
        mlrun.utils.logger.warn(
            f"additional_filters parameter is not supported in {class_name},"
            f" parameter has been ignored."
        )


def merge_dicts_with_precedence(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries with precedence given to keys from later dictionaries.

    This function merges an arbitrary number of dictionaries, where keys from dictionaries later
    in the argument list take precedence over keys from dictionaries earlier in the list. If all
    dictionaries contain the same key, the value from the last dictionary with that key will
    overwrite the values from earlier dictionaries.

    Example:
        >>> first_dict = {"key1": "value1", "key2": "value2"}
        >>> second_dict = {"key2": "new_value2", "key3": "value3"}
        >>> third_dict = {"key3": "new_value3", "key4": "value4"}
        >>> merge_dicts_with_precedence(first_dict, second_dict, third_dict)
        {'key1': 'value1', 'key2': 'new_value2', 'key3': 'new_value3', 'key4': 'value4'}

    - If no dictionaries are provided, the function returns an empty dictionary.
    """
    return {k: v for d in dicts if d for k, v in d.items()}


def validate_component_version_compatibility(
    component_name: typing.Literal["iguazio", "nuclio", "mlrun-client"],
    *min_versions: str,
    mlrun_client_version: Optional[str] = None,
):
    """
    :param component_name: Name of the component to validate compatibility for.
    :param min_versions: Valid minimum version(s) required, assuming no 2 versions has equal major and minor.
    :param mlrun_client_version: Client version to validate when component_name is "mlrun-client".
    """
    parsed_min_versions = [
        semver.VersionInfo.parse(min_version) for min_version in min_versions
    ]
    parsed_current_version = None
    component_current_version = None
    # For mlrun client we don't assume compatability if we fail to parse the client version
    assume_compatible = component_name not in ["mlrun-client"]
    try:
        if component_name == "iguazio":
            component_current_version = mlrun.mlconf.igz_version
            parsed_current_version = mlrun.mlconf.get_parsed_igz_version()

            if parsed_current_version:
                # ignore pre-release and build metadata, as iguazio version always has them, and we only care about the
                # major, minor, and patch versions
                parsed_current_version = semver.VersionInfo.parse(
                    f"{parsed_current_version.major}.{parsed_current_version.minor}.{parsed_current_version.patch}"
                )
        if component_name == "nuclio":
            component_current_version = mlrun.mlconf.nuclio_version
            parsed_current_version = semver.VersionInfo.parse(
                mlrun.mlconf.nuclio_version
            )
        if component_name == "mlrun-client":
            # dev version, assume compatible
            if mlrun_client_version and (
                mlrun_client_version.startswith("0.0.0+")
                or "unstable" in mlrun_client_version
            ):
                return True

            component_current_version = mlrun_client_version
            parsed_current_version = semver.Version.parse(mlrun_client_version)
        if not parsed_current_version:
            return assume_compatible
    except ValueError:
        # only log when version is set but invalid
        if component_current_version:
            logger.warning(
                "Unable to parse current version",
                component_name=component_name,
                current_version=component_current_version,
                min_versions=min_versions,
                assume_compatible=assume_compatible,
            )
        return assume_compatible

    # Feature might have been back-ported e.g. nuclio node selection is supported from
    # 1.5.20 and 1.6.10 but not in 1.6.9 - therefore we reverse sort to validate against 1.6.x 1st and
    # then against 1.5.x
    parsed_min_versions.sort(reverse=True)
    for parsed_min_version in parsed_min_versions:
        if (
            parsed_current_version.major == parsed_min_version.major
            and parsed_current_version.minor == parsed_min_version.minor
            and parsed_current_version.patch < parsed_min_version.patch
        ):
            return False

        if parsed_current_version >= parsed_min_version:
            return True
    return False


def format_alert_summary(
    alert: mlrun.common.schemas.AlertConfig, event_data: mlrun.common.schemas.Event
) -> str:
    result = alert.summary.replace("{{project}}", alert.project)
    result = result.replace("{{name}}", alert.name)
    result = result.replace("{{entity}}", event_data.entity.ids[0])
    return result


def is_parquet_file(file_path, format_=None):
    return (file_path and file_path.endswith((".parquet", ".pq"))) or (
        format_ == "parquet"
    )


def validate_single_def_handler(function_kind: str, code: str):
    # The name of MLRun's wrapper is 'handler', which is why the handler function name cannot be 'handler'
    # it would override MLRun's wrapper
    if function_kind == "mlrun":
        # Find all lines that start with "def handler("
        pattern = re.compile(r"^def handler\(", re.MULTILINE)
        matches = pattern.findall(code)

        # Only MLRun's wrapper handler (footer) can be in the code
        if len(matches) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The code file contains a function named “handler“, which is reserved. "
                + "Use a different name for your function."
            )


def _reload(module, max_recursion_depth):
    """Recursively reload modules."""
    if max_recursion_depth <= 0:
        return

    reload(module)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType:
            _reload(attribute, max_recursion_depth - 1)


def run_with_retry(
    retry_count: int,
    func: typing.Callable,
    retry_on_exceptions: Optional[
        typing.Union[type[Exception], tuple[type[Exception]]]
    ] = None,
    *args,
    **kwargs,
):
    """
    Executes a function with retry logic upon encountering specified exceptions.

    :param retry_count: The number of times to retry the function execution.
    :param func: The function to execute.
    :param retry_on_exceptions: Exception(s) that trigger a retry. Can be a single exception or a tuple of exceptions.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function execution if successful.
    :raises Exception: Re-raises the last exception encountered after all retries are exhausted.
    """
    if retry_on_exceptions is None:
        retry_on_exceptions = (Exception,)
    elif isinstance(retry_on_exceptions, list):
        retry_on_exceptions = tuple(retry_on_exceptions)

    last_exception = None
    for attempt in range(retry_count + 1):
        try:
            return func(*args, **kwargs)
        except retry_on_exceptions as exc:
            last_exception = exc
            logger.warning(
                f"Attempt {{{attempt}/ {retry_count}}} failed with exception: {exc}",
            )
            if attempt == retry_count:
                raise
    raise last_exception


def join_urls(base_url: Optional[str], path: Optional[str]) -> str:
    """
    Joins a base URL with a path, ensuring proper handling of slashes.

    :param base_url: The base URL (e.g., "http://example.com").
    :param path: The path to append to the base URL (e.g., "/path/to/resource").

    :return: A unified URL with exactly one slash between base_url and path.
    """
    if base_url is None:
        base_url = ""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}" if path else base_url


class Workflow:
    @staticmethod
    def get_workflow_steps(
        db: "mlrun.db.RunDBInterface", workflow_id: str, project: str
    ) -> list:
        steps = []

        def _add_run_step(_step: mlrun_pipelines.models.PipelineStep):
            # on kfp 1.8 argo sets the pod hostname differently than what we have with kfp 2.5
            # therefore, the heuristic needs to change. what we do here is first trying against 1.8 conventions
            # and if we can't find it then falling back to 2.5
            try:
                # runner_pod = x-y-N
                _runs = db.list_runs(
                    project=project,
                    labels=f"{mlrun_constants.MLRunInternalLabels.runner_pod}={_step.node_name}",
                )
                if not _runs:
                    try:
                        # x-y-N -> x-y, N
                        node_name_initials, node_name_generated_id = (
                            _step.node_name.rsplit("-", 1)
                        )

                    except ValueError:
                        # defensive programming, if the node name is not in the expected format
                        node_name_initials = _step.node_name
                        node_name_generated_id = ""

                    # compile the expected runner pod hostname as per kfp >= 2.4
                    # x-y, Z, N -> runner_pod = x-y-Z-N
                    runner_pod_value = "-".join(
                        [
                            node_name_initials,
                            _step.display_name,
                            node_name_generated_id,
                        ]
                    ).rstrip("-")
                    logger.debug(
                        "No run found for step, trying with different node name",
                        step_node_name=runner_pod_value,
                    )
                    _runs = db.list_runs(
                        project=project,
                        labels=f"{mlrun_constants.MLRunInternalLabels.runner_pod}={runner_pod_value}",
                    )

                _run = _runs[0]
            except IndexError:
                logger.warning("No run found for step", step=_step.to_dict())
                _run = {
                    "metadata": {
                        "name": _step.display_name,
                        "project": project,
                    },
                    "status": {},
                }
            _run["step_kind"] = _step.step_type
            if _step.skipped:
                _run.setdefault("status", {})["state"] = (
                    runtimes_constants.RunStates.skipped
                )
            steps.append(_run)

        def _add_deploy_function_step(_step: mlrun_pipelines.models.PipelineStep):
            project, name, hash_key = Workflow._extract_function_uri(
                _step.get_annotation("mlrun/function-uri")
            )
            if name:
                try:
                    function = db.get_function(
                        project=project, name=name, hash_key=hash_key
                    )
                except mlrun.errors.MLRunNotFoundError:
                    # If the function is not found (if build failed for example), we will create a dummy
                    # function object for the notification to display the function name
                    function = {
                        "metadata": {
                            "name": name,
                            "project": project,
                            "hash_key": hash_key,
                        },
                    }
                pod_phase = _step.phase
                if _step.skipped:
                    state = mlrun.common.schemas.FunctionState.skipped
                else:
                    state = runtimes_constants.PodPhases.pod_phase_to_run_state(
                        pod_phase
                    )
                function["status"] = {"state": state}
                if isinstance(function["metadata"].get("updated"), datetime):
                    function["metadata"]["updated"] = function["metadata"][
                        "updated"
                    ].isoformat()
                function["step_kind"] = _step.step_type
                steps.append(function)

        step_methods = {
            mlrun_pipelines.common.constants.PipelineRunType.run: _add_run_step,
            mlrun_pipelines.common.constants.PipelineRunType.build: _add_deploy_function_step,
            mlrun_pipelines.common.constants.PipelineRunType.deploy: _add_deploy_function_step,
        }

        if not workflow_id:
            return steps

        try:
            workflow_manifest = Workflow._get_workflow_manifest(workflow_id)
        except Exception:
            logger.warning(
                "Failed to extract workflow steps from workflow manifest, "
                "returning all runs with the workflow id label",
                workflow_id=workflow_id,
                traceback=traceback.format_exc(),
            )
            return db.list_runs(
                project=project,
                labels=f"workflow={workflow_id}",
            )

        if not workflow_manifest:
            return steps

        try:
            for step in workflow_manifest.get_steps():
                step_method = step_methods.get(step.step_type)
                if step_method:
                    step_method(step)
            return steps
        except Exception:
            # If we fail to read the pipeline steps, we will return the list of runs that have the same workflow id
            logger.warning(
                "Failed to extract workflow steps from workflow manifest, "
                "returning all runs with the workflow id label",
                workflow_id=workflow_id,
                traceback=traceback.format_exc(),
            )
            return db.list_runs(
                project=project,
                labels=f"workflow={workflow_id}",
            )

    @staticmethod
    def _extract_function_uri(function_uri: str) -> tuple[str, str, str]:
        """
        Extract the project, name, and hash key from a function uri.
        Examples:
            - "project/name@hash_key" returns project, name, hash_key
            - "project/name returns" project, name, ""
        """
        project, name, hash_key = None, None, None
        hashed_pattern = r"^(.+)/(.+)@(.+)$"
        pattern = r"^(.+)/(.+)$"
        match = re.match(hashed_pattern, function_uri)
        if match:
            project, name, hash_key = match.groups()
        else:
            match = re.match(pattern, function_uri)
            if match:
                project, name = match.groups()
                hash_key = ""
        return project, name, hash_key

    @staticmethod
    def _get_workflow_manifest(
        workflow_id: str,
    ) -> typing.Optional[mlrun_pipelines.models.PipelineManifest]:
        kfp_client = mlrun_pipelines.utils.get_client(
            url=mlrun.mlconf.kfp_url,
            namespace=mlrun.mlconf.namespace,
        )

        # arbitrary timeout of 30 seconds, the workflow should be done by now, however sometimes kfp takes a few
        # seconds to update the workflow status
        kfp_run = kfp_client.wait_for_run_completion(workflow_id, 30)
        if not kfp_run:
            return None

        kfp_run = mlrun_pipelines.models.PipelineRun(kfp_run)
        return kfp_run.workflow_manifest()


def as_dict(data: typing.Union[dict, str]) -> dict:
    if isinstance(data, str):
        return json.loads(data)
    return data


def encode_user_code(
    user_code: typing.Union[str, bytes], max_len_warning: typing.Optional[int] = None
) -> str:
    max_len_warning = max_len_warning or config.function.spec.source_code_max_bytes
    if isinstance(user_code, str):
        user_code = user_code.encode("utf-8")
    encoded = base64.b64encode(user_code).decode("utf-8")
    if len(encoded) > max_len_warning:
        logger.warning(
            f"User code exceeds the maximum allowed size of {max_len_warning} bytes for non remote source. "
            "Consider using `with_source_archive` to add user code as a remote source to the function."
        )
    return encoded
