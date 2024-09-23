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
#
from dateutil import parser

import mlrun.common.runtimes.constants
from mlrun.utils import get_in
from server.api.db.sqldb.models import Base

max_str_length = 255


def label_set(labels):
    if isinstance(labels, str):
        labels = labels.split(",")

    return set(labels or [])


def transform_label_list_to_dict(label_list):
    return {label.name: label.value for label in label_list}


def run_start_time(run):
    ts = get_in(run, "status.start_time", "")
    if not ts:
        return None
    return parser.parse(ts)


def run_labels(run) -> dict:
    return get_in(run, "metadata.labels", {})


def run_state(run):
    return get_in(
        run, "status.state", mlrun.common.runtimes.constants.RunStates.created
    )


def update_labels(obj, labels: dict):
    old = {label.name: label for label in obj.labels}
    obj.labels.clear()
    for name, value in labels.items():
        if name in old:
            old[name].value = value
            obj.labels.append(old[name])
        else:
            obj.labels.append(obj.Label(name=name, value=value, parent=obj.id))


def to_dict(obj):
    if isinstance(obj, Base):
        return {
            attr: to_dict(getattr(obj, attr)) for attr in dir(obj) if is_field(attr)
        }

    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(to_dict(v) for v in obj)

    return obj


def is_field(name):
    if name[0] == "_":
        return False
    return name not in ("metadata", "Tag", "Label", "body")


def generate_query_predicate_for_name(column, query_string):
    if query_string.startswith("~"):
        return column.ilike(f"%{query_string[1:]}%")
    else:
        return column.__eq__(query_string)


def ensure_max_length(string: str):
    if string and len(string) > max_str_length:
        string = string[:max_str_length]
    return string


class MemoizationCache:
    _not_found_object = object()

    def __init__(self, function):
        self._function = function
        self._cache = {}

    def memoize(self, *args, **kwargs):
        # kwargs are not included in the memoization key
        memo_key = tuple(id(arg) for arg in args)
        result = self._cache.get(memo_key, self._not_found_object)
        if result is self._not_found_object:
            result = self._function(*args, **kwargs)
            self._cache[memo_key] = result
        return result
