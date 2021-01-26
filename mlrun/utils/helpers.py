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
import re
import sys
import time
from types import ModuleType
from typing import Optional, Tuple
from datetime import datetime, timezone
from dateutil import parser
from os import path, environ
from importlib import import_module
import inspect

import numpy as np
import requests
import yaml
from pandas._libs.tslibs.timestamps import Timestamp
from tabulate import tabulate
from yaml.representer import RepresenterError

import mlrun.utils.version.version
import mlrun.errors
from .logger import create_logger
from ..config import config

yaml.Dumper.ignore_aliases = lambda *args: True
_missing = object()

hub_prefix = "hub://"
DB_SCHEMA = "store"


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
    kind = item.get("kind")
    if kind in ["dataset", "model"] and item.get("db_key"):
        return "{}://{}/{}/{}:{}".format(
            DB_SCHEMA,
            StorePrefix.Artifact,
            project or item.get("project"),
            item.get("db_key"),
            item.get("tree"),
        )
    return item.get("target_path")


logger = create_logger(config.log_level, config.log_formatter, "mlrun", sys.stdout)
missing = object()

is_ipython = False
try:
    import IPython

    ipy = IPython.get_ipython()
    if ipy:
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


def verify_field_regex(field_name, field_value, patterns):
    logger.debug(
        "Validating field against patterns",
        field_name=field_name,
        field_value=field_value,
        pattern=patterns,
    )

    for pattern in patterns:
        if not re.match(pattern, str(field_value)):
            logger.warn(
                "Field is malformed. Does not match required pattern",
                field_name=field_name,
                field_value=field_value,
                pattern=pattern,
            )
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Field {field_name} is malformed. Does not match required pattern: {pattern}"
            )


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


def update_in(obj, key, value, append=False, replace=True):
    parts = key.split(".") if isinstance(key, str) else key
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
            raise ValueError("illegal condition - {}".format(text))
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


def flatten(df, col, prefix=""):
    params = []
    for r in df[col]:
        if r:
            for k in r.keys():
                if k not in params:
                    params += [k]
    params
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
    return ["{}={}".format(k, v) for k, v in struct.items()]


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


yaml.add_representer(np.int64, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.integer, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.floating, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.ndarray, numpy_representer_seq, Dumper=yaml.SafeDumper)
yaml.add_representer(np.datetime64, date_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(Timestamp, date_representer, Dumper=yaml.SafeDumper)


def dict_to_yaml(struct):
    try:
        data = yaml.safe_dump(struct, default_flow_style=False, sort_keys=False)
    except RepresenterError as e:
        raise ValueError(
            "error: data result cannot be serialized to YAML" ", {} ".format(e)
        )
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


def uxjoin(base, local_path, key="", iter=None, is_dir=False):
    if is_dir and (not local_path or local_path in [".", "./"]):
        local_path = ""
    elif not local_path:
        local_path = key

    if iter:
        local_path = path.join(str(iter), local_path)

    if base and not base.endswith("/"):
        base += "/"
    return "{}{}".format(base or "", local_path)


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


def generate_object_uri(project, name, tag=None, hash_key=None):
    uri = "{}/{}".format(project, name)

    # prioritize hash key over tag
    if hash_key:
        uri += "@{}".format(hash_key)
    elif tag:
        uri += ":{}".format(tag)
    return uri


def extend_hub_uri(uri):
    if not uri.startswith(hub_prefix):
        return uri
    name = uri[len(hub_prefix) :]
    tag = "master"
    if ":" in name:
        loc = name.find(":")
        tag = name[loc + 1 :]
        name = name[:loc]

    # hub function directory name are with underscores instead of hyphens
    name = name.replace("-", "_")
    return config.hub_url.format(name=name, tag=tag)


def gen_md_table(header, rows=None):
    rows = [] if rows is None else rows

    def gen_list(items=None):
        items = [] if items is None else items
        out = "|"
        for i in items:
            out += " {} |".format(i)
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
            out += "<{}>{}</{}>".format(tag, item, tag)
        return out

    out = "<tr>" + gen_list(header, "th") + "</tr>\n"
    for r in rows:
        out += "<tr>" + gen_list(r, "td") + "</tr>\n"
    return style + '<table class="tg">\n' + out + "</table>\n\n"


def new_pipe_meta(artifact_path=None, ttl=None, *args):
    from kfp.dsl import PipelineConf

    def _set_artifact_path(task):
        from kubernetes import client as k8s_client

        task.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_ARTIFACT_PATH", value=artifact_path)
        )
        return task

    conf = PipelineConf()
    ttl = ttl or int(config.kfp_ttl)
    if ttl:
        conf.set_ttl_seconds_after_finished(ttl)
    if artifact_path:
        conf.add_op_transformer(_set_artifact_path)
    for op in args:
        if op:
            conf.add_op_transformer(op)
    return conf


def enrich_image_url(image_url: str) -> str:
    tag = config.images_tag or mlrun.utils.version.Version().get()["version"]
    registry = config.images_registry
    if image_url.startswith("mlrun/") or "/mlrun/" in image_url:
        if tag and ":" not in image_url:
            image_url = f"{image_url}:{tag}"
        if registry and "/mlrun/" not in image_url:
            image_url = f"{registry}{image_url}"
    return image_url


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


def pr_comment(repo: str, issue: int, message: str, token=None):
    token = token or environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    url = f"https://api.github.com/repos/{repo}/issues/{issue}/comments"

    resp = requests.post(url=url, json={"body": str(message)}, headers=headers)
    if not resp.ok:
        errmsg = f"bad pr comment resp!!\n{resp.text}"
        raise IOError(errmsg)
    return resp.json()["id"]


def fill_object_hash(object_dict, uid_property_name, tag=""):
    # remove tag, hash, date from calculation
    object_dict.setdefault("metadata", {})
    tag = tag or object_dict["metadata"].get("tag")
    status = object_dict.setdefault("status", {})
    object_dict["metadata"]["tag"] = ""
    object_dict["metadata"][uid_property_name] = ""
    object_dict["status"] = None
    object_dict["metadata"]["updated"] = None
    data = json.dumps(object_dict, sort_keys=True).encode()
    h = hashlib.sha1()
    h.update(data)
    uid = h.hexdigest()
    object_dict["metadata"]["tag"] = tag
    object_dict["metadata"][uid_property_name] = uid
    object_dict["status"] = status
    return uid


def fill_function_hash(function_dict, tag=""):
    return fill_object_hash(function_dict, "hash", tag)


def retry_until_successful(
    interval: int, timeout: int, logger, verbose: bool, _function, *args, **kwargs
):
    """
    Runs function with given *args and **kwargs.
    Tries to run it until success or timeout reached (timeout is optional)
    :param interval: int/float that will be used as interval
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

    # If deadline was not provided or deadline not reached
    while timeout is None or time.time() < start_time + timeout:
        try:
            result = _function(*args, **kwargs)
            return result

        except Exception as exc:
            last_exception = exc

            # If next interval is within allowed time period - wait on interval, abort otherwise
            if timeout is None or time.time() + interval < start_time + timeout:
                if logger is not None and verbose:
                    logger.debug(
                        f"Operation not yet successful, Retrying in {interval} seconds. exc: {exc}"
                    )

                time.sleep(interval)
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


class RunNotifications:
    def __init__(self, with_ipython=True, with_slack=False):
        self._hooks = []
        self._html = ""
        self.with_ipython = with_ipython
        if with_slack and "SLACK_WEBHOOK" in environ:
            self.slack()

    def push(self, message, runs):
        for h in self._hooks:
            try:
                h(message, runs)
            except Exception as e:
                logger.warning(f"failed to push notification, {e}")
        if self.with_ipython and is_ipython:
            import IPython

            IPython.display.display(IPython.display.HTML(self._get_html(message, runs)))

    def _get_html(self, message, runs):
        if self._html:
            return self._html

        html = "<h2>Run Results</h2>" + message
        html += "<br>click the hyper links below to see detailed results<br>"
        html += runs.show(display=False, short=True)
        self._html = html
        return html

    def print(self):
        def _print(message, runs):
            table = []
            for r in runs:
                state = r["status"].get("state", "")
                if state == "error":
                    result = r["status"].get("error", "")
                else:
                    result = dict_to_str(r["status"].get("results", {}))

                table.append(
                    [
                        state,
                        r["metadata"]["name"],
                        ".." + r["metadata"]["uid"][-6:],
                        result,
                    ]
                )
            print(
                message
                + "\n"
                + tabulate(table, headers=["status", "name", "uid", "results"])
            )

        self._hooks.append(_print)
        return self

    def slack(self, webhook=""):
        emoji = {"completed": ":smiley:", "running": ":man-running:", "error": ":x:"}

        template = "{}/{}/{}/jobs/{}/info"

        webhook = webhook or environ.get("SLACK_WEBHOOK")
        if not webhook:
            raise ValueError("Slack webhook is not set")

        def row(text):
            return {"type": "mrkdwn", "text": text}

        def _slack(message, runs):
            fields = [row("*Runs*"), row("*Results*")]
            for r in runs:
                meta = r["metadata"]
                if config.resolve_ui_url():
                    url = template.format(
                        config.resolve_ui_url(),
                        config.ui.projects_prefix,
                        meta.get("project"),
                        meta.get("uid"),
                    )
                    line = f'<{url}|*{meta.get("name")}*>'
                else:
                    line = meta.get("name")
                state = r["status"].get("state", "")
                line = f'{emoji.get(state, ":question:")}  {line}'

                fields.append(row(line))
                if state == "error":
                    result = "*{}*".format(r["status"].get("error", ""))
                else:
                    result = dict_to_str(r["status"].get("results", {}), ", ")
                fields.append(row(result or "None"))

            data = {
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": message}}
                ]
            }

            for i in range(0, len(fields), 8):
                data["blocks"].append({"type": "section", "fields": fields[i : i + 8]})
            response = requests.post(
                webhook,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        self._hooks.append(_slack)
        return self

    def git_comment(self, git_repo=None, git_issue=None, token=None):
        def _comment(message, runs):
            pr_comment(
                git_repo or self._get_param("git_repo"),
                git_issue or self._get_param("git_issue"),
                self._get_html(message, runs),
                token=token,
            )

        self._hooks.append(_comment)
        return self


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


def get_class(class_name, namespace):
    """return class object from class name string"""
    if isinstance(class_name, type):
        return class_name
    namespace = _module_to_namespace(namespace)
    if namespace and class_name in namespace:
        return namespace[class_name]

    try:
        class_object = create_class(class_name)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, class {class_name} not found, {e}")
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
    namespace = _module_to_namespace(namespace)
    if function in namespace:
        return namespace[function]

    try:
        function_object = create_function(function)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, function {function} not found, {e}")
    return function_object


def datetime_from_iso(time_str: str) -> Optional[datetime]:
    if not time_str:
        return
    return parser.isoparse(time_str)


def datetime_to_iso(time_obj: Optional[datetime]) -> Optional[str]:
    if not time_obj:
        return
    return time_obj.isoformat()
