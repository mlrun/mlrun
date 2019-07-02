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

import json
from os import path
import yaml


class run_keys:
    input_path = 'default_input_path'
    output_path = 'default_output_path'
    input_objects = 'input_objects'
    output_artifacts = 'output_artifacts'
    data_stores = 'data_stores'
    secrets = 'secret_sources'


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


def uxjoin(base, path):
    if base:
        if not base.endswith('/'):
            base += '/'
        return '{}{}'.format(base, path)
    return path


class ModelObj:
    _dict_fields = []

    def to_dict(self, fields=None):
        struct = {}
        fields = fields or self._dict_fields
        for t in fields:
            val = getattr(self, t, None)
            if val is not None:
                if hasattr(val, 'to_dict'):
                    struct[t] = val.to_dict()
                else:
                    struct[t] = val
        return struct

    #@classmethod
    #def from_dict(cls, struct={}):
    #    for key, val in struct.items():

    def to_yaml(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self):
        return json.dumps(self.to_dict())

