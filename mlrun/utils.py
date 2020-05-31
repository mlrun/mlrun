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

import time
import hashlib
import json
import logging
import pathlib
import re
from datetime import datetime, timezone
from os import path, environ
from sys import stdout

import numpy as np
import requests
import yaml
from tabulate import tabulate
from yaml.representer import RepresenterError

from .config import config

yaml.Dumper.ignore_aliases = lambda *args: True
_missing = object()

hub_prefix = 'hub://'
DB_SCHEMA = 'store'


def create_logger(stream=None):
    level = logging.INFO
    if config.log_level.lower() == 'debug':
        level = logging.DEBUG
    handler = logging.StreamHandler(stream or stdout)
    handler.setFormatter(
        logging.Formatter('[%(name)s] %(asctime)s %(message)s'))
    handler.setLevel(level)
    logger = logging.getLogger('mlrun')
    if not len(logger.handlers):
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = create_logger()
missing = object()

is_ipython = False
try:
    import IPython

    ipy = IPython.get_ipython()
    if ipy:
        is_ipython = True
except ImportError:
    pass

if is_ipython:
    # bypass Jupyter asyncio bug
    import nest_asyncio

    nest_asyncio.apply()


class run_keys:
    input_path = 'input_path'
    output_path = 'output_path'
    inputs = 'inputs'
    artifacts = 'artifacts'
    outputs = 'outputs'
    data_stores = 'data_stores'
    secrets = 'secret_sources'


def now_date():
    return datetime.now(timezone.utc)


def to_date_str(d):
    if d:
        return d.isoformat()
    return ''


def normalize_name(name):
    # TODO: Must match
    # [a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?
    name = re.sub(r'\s+', '-', name)
    name = name.replace('_', '-')
    return name.lower()


class LogBatchWriter:
    def __init__(self, func, batch=16, maxtime=5):
        self.batch = batch
        self.maxtime = maxtime
        self.start_time = datetime.now()
        self.buffer = ''
        self.func = func

    def write(self, data):
        self.buffer += data
        self.batch -= 1
        elapsed_time = (datetime.now() - self.start_time).seconds
        if elapsed_time > self.maxtime or self.batch <= 0:
            self.flush()

    def flush(self):
        self.func(self.buffer)
        self.buffer = ''
        self.start_time = datetime.now()


def get_in(obj, keys, default=None):
    """
    >>> get_in({'a': {'b': 1}}, 'a.b')
    1
    """
    if isinstance(keys, str):
        keys = keys.split('.')

    for key in keys:
        if not obj or key not in obj:
            return default
        obj = obj[key]
    return obj


def update_in(obj, key, value, append=False, replace=True):
    parts = key.split('.') if isinstance(key, str) else key
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
            raise ValueError('illegal condition - {}'.format(text))
        return labels.get(items[0].strip(), ''), items[1].strip()

    for condition in conditions:
        if '~=' in condition:
            l, val = splitter('~=', condition)
            match = match and val in l
        elif '!=' in condition:
            l, val = splitter('!=', condition)
            match = match and val != l
        elif '=' in condition:
            l, val = splitter('=', condition)
            match = match and val == l
        else:
            match = match and (condition.strip() in labels)
    return match


def match_value(value, obj, key):
    if not value:
        return True
    return get_in(obj, key, _missing) == value


def flatten(df, col, prefix=''):
    params = []
    for r in df[col]:
        if r:
            for k in r.keys():
                if k not in params:
                    params += [k]
    params
    for p in params:
        df[prefix + p] = df[col].apply(lambda x: x.get(p, '') if x else '')
    df.drop(col, axis=1, inplace=True)
    return df


def list2dict(lines: list):
    out = {}
    for line in lines:
        i = line.find('=')
        if i == -1:
            continue
        key, value = line[:i].strip(), line[i + 1:].strip()
        if key is None:
            raise ValueError('cannot find key in line (key=value)')
        value = path.expandvars(value)
        out[key] = value
    return out


def dict_to_list(struct: dict):
    if not struct:
        return []
    return ['{}={}'.format(k, v) for k, v in struct.items()]


def dict_to_str(struct: dict, sep=','):
    return sep.join(dict_to_list(struct))


def numpy_representer_seq(dumper, data):
    return dumper.represent_list(data.tolist())


def float_representer(dumper, data):
    return dumper.represent_float(data)


def int_representer(dumper, data):
    return dumper.represent_int(data)


yaml.add_representer(np.int64, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.integer, int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.floating, float_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.ndarray, numpy_representer_seq, Dumper=yaml.SafeDumper)


def dict_to_yaml(struct):
    try:
        data = yaml.safe_dump(struct, default_flow_style=False,
                              sort_keys=False)
    except RepresenterError as e:
        raise ValueError('error: data result cannot be serialized to YAML'
                         ', {} '.format(e))
    return data


# solve numpy json serialization
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pathlib.PosixPath):
            return str(obj)
        elif np.isnan(obj) or np.isinf(obj):
            return str(obj)
        else:
            return super(MyEncoder, self).default(obj)


def dict_to_json(struct):
    return json.dumps(struct, cls=MyEncoder)


def uxjoin(base, local_path, key='', iter=None, is_dir=False):
    if is_dir and (not local_path or local_path in ['.', './']):
        local_path = ''
    elif not local_path:
        local_path = key

    if iter:
        local_path = path.join(str(iter), local_path)

    if base and not base.endswith('/'):
        base += '/'
    return '{}{}'.format(base or '', local_path)


def parse_function_uri(uri):
    project = ''
    tag = ''
    hash_key = ''
    if '/' in uri:
        loc = uri.find('/')
        project = uri[:loc]
        uri = uri[loc + 1:]
    if ':' in uri:
        loc = uri.find(':')
        tag = uri[loc + 1:]
        uri = uri[:loc]
    if '@' in uri:
        loc = uri.find('@')
        hash_key = uri[loc + 1:]
        uri = uri[:loc]
    return project, uri, tag, hash_key


def extend_hub_uri(uri):
    if not uri.startswith(hub_prefix):
        return uri
    name = uri[len(hub_prefix):]
    tag = 'master'
    if ':' in name:
        loc = name.find(':')
        tag = name[loc + 1:]
        name = name[:loc]
    return config.hub_url.format(name=name, tag=tag)


def gen_md_table(header, rows=None):
    rows = [] if rows is None else rows

    def gen_list(items=None):
        items = [] if items is None else items
        out = '|'
        for i in items:
            out += ' {} |'.format(i)
        return out

    out = gen_list(header) + '\n' + gen_list(len(header) * ['---']) + '\n'
    for r in rows:
        out += gen_list(r) + '\n'
    return out


def gen_html_table(header, rows=None):
    rows = [] if rows is None else rows

    style = '''
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-style:solid;border-width:1px;padding:6px 4px;}
.tg th{font-weight:normal;border-style:solid;border-width:1px;padding:6px 4px;}
</style>
'''

    def gen_list(items=None, tag='td'):
        items = [] if items is None else items
        out = ''
        for item in items:
            out += '<{}>{}</{}>'.format(tag, item, tag)
        return out

    out = '<tr>' + gen_list(header, 'th') + '</tr>\n'
    for r in rows:
        out += '<tr>' + gen_list(r, 'td') + '</tr>\n'
    return style + '<table class="tg">\n' + out + '</table>\n\n'


def new_pipe_meta(artifact_path=None, ttl=None, *args):
    from kfp.dsl import PipelineConf

    def _set_artifact_path(task):
        from kubernetes import client as k8s_client
        task.add_env_variable(k8s_client.V1EnvVar(
            name='MLRUN_ARTIFACT_PATH', value=artifact_path))
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


def tag_image(base: str):
    ver = config.images_tag or config.version
    if ver and (base == 'mlrun/mlrun' or (
            base.startswith('mlrun/ml-') and ':' not in base)):
        base += ':' + ver
    return base


def get_artifact_target(item: dict, project=None):
    kind = item.get('kind')
    if kind in ['dataset', 'model'] and item.get('db_key'):
        return '{}://{}/{}#{}'.format(DB_SCHEMA,
                                      project or item.get('project'),
                                      item.get('db_key'), item.get('tree'))
    return item.get('target_path')


def pr_comment(repo: str, issue: int,
               message: str, token=None):
    token = token or environ.get('GITHUB_TOKEN')
    headers = {'Accept': 'application/vnd.github.v3+json',
               'Authorization': f'token {token}'}
    url = f'https://api.github.com/repos/{repo}/issues/{issue}/comments'

    resp = requests.post(url=url, json={"body": str(message)}, headers=headers)
    if not resp.ok:
        errmsg = f'bad pr comment resp!!\n{resp.text}'
        raise IOError(errmsg)
    return resp.json()['id']


def fill_function_hash(function_dict, tag=''):

    # remove tag, hash, date from calculation
    function_dict.setdefault('metadata', {})
    tag = tag or function_dict['metadata'].get('tag')
    status = function_dict.setdefault('status', {})
    function_dict['metadata']['tag'] = ''
    function_dict['metadata']['hash'] = ''
    function_dict['status'] = None
    function_dict['metadata']['updated'] = None

    data = json.dumps(function_dict, sort_keys=True).encode()
    h = hashlib.sha1()
    h.update(data)
    hashkey = h.hexdigest()
    function_dict['metadata']['tag'] = tag
    function_dict['metadata']['hash'] = hashkey
    function_dict['status'] = status
    return hashkey


def retry_until_successful(interval: int, timeout: int, logger, verbose: bool, _function, *args, **kwargs):
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
                    logger.debug(f"Operation not yet successful, Retrying in {interval} seconds. exc: {exc}")

                time.sleep(interval)
            else:
                break

    if logger is not None:
        logger.warning(f"Operation did not complete on time. last exception: {last_exception}")

    raise Exception(f"failed to execute command by the given deadline."
                    f" last_exception: {last_exception},"
                    f" function_name: {_function.__name__},"
                    f" timeout: {timeout}")


class RunNotifications:
    def __init__(self, with_ipython=True, with_slack=False):
        self._hooks = []
        self._html = ''
        self.with_ipython = with_ipython
        if with_slack and 'SLACK_WEBHOOK' in environ:
            self.slack()

    def push(self, message, runs):
        for h in self._hooks:
            try:
                h(message, runs)
            except Exception as e:
                logger.warning(f'failed to push notification, {e}')
        if self.with_ipython and is_ipython:
            import IPython
            IPython.display.display(IPython.display.HTML(
                self._get_html(message, runs)))

    def _get_html(self, message, runs):
        if self._html:
            return self._html

        html = '<h2>Run Results</h2>' + message
        html += '<br>click the hyper links below to see detailed results<br>'
        html += runs.show(display=False, short=True)
        self._html = html
        return html

    def print(self):
        def _print(message, runs):
            table = []
            for r in runs:
                state = r['status'].get('state', '')
                if state == 'error':
                    result = r['status'].get('error', '')
                else:
                    result = dict_to_str(r['status'].get('results', {}))

                table.append([state,
                              r['metadata']['name'],
                              '..' + r['metadata']['uid'][-6:],
                              result])
            print(message + '\n' + tabulate(
                table, headers=['status', 'name', 'uid', 'results']))

        self._hooks.append(_print)
        return self

    def slack(self, webhook=''):
        emoji = {'completed': ':smiley:',
                 'running': ':man-running:',
                 'error': ':x:'}

        template = '{}/projects/{}/jobs/{}/info'

        webhook = webhook or environ.get('SLACK_WEBHOOK')
        if not webhook:
            raise ValueError('Slack webhook is not set')

        def row(text):
            return {'type': 'mrkdwn', 'text': text}

        def _slack(message, runs):
            fields = [row('*Runs*'), row('*Results*')]
            for r in runs:
                meta = r['metadata']
                if config.ui_url:
                    url = template.format(config.ui_url, meta.get('project'), meta.get('uid'))
                    line = f'<{url}|*{meta.get("name")}*>'
                else:
                    line = meta.get("name")
                state = r['status'].get('state', '')
                line = f'{emoji.get(state, ":question:")}  {line}'

                fields.append(row(line))
                if state == 'error':
                    result = '*{}*'.format(r['status'].get('error', ''))
                else:
                    result = dict_to_str(r['status'].get('results', {}), ', ')
                fields.append(row(result or 'None'))

            data = {
                'blocks': [
                    {"type": "section",
                     "text": {"type": "mrkdwn", "text": message}
                     }
                ]
            }

            for i in range(0, len(fields), 8):
                data['blocks'].append({"type": "section",
                                       "fields": fields[i:i + 8]})
            response = requests.post(webhook, data=json.dumps(
                data), headers={'Content-Type': 'application/json'})
            response.raise_for_status()

        self._hooks.append(_slack)
        return self

    def git_comment(self, git_repo=None, git_issue=None, token=None):
        def _comment(message, runs):
            pr_comment(git_repo or self._get_param('git_repo'),
                       git_issue or self._get_param('git_issue'),
                       self._get_html(message, runs),
                       token=token)

        self._hooks.append(_comment)
        return self


