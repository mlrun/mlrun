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

import copy
import typing
import yaml

from .loader import ConfigLoader


class ConfigBase(object):

    _missing = object()
    dynamic_attributes = []

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg

        # Can't use self._cfg = cfg → infinite recursion
        super().__setattr__("_cfg", cfg)

    def __getattr__(self, attr):
        val = self._cfg.get(attr, self._missing)
        if val is self._missing:
            raise AttributeError(attr)

        if isinstance(val, typing.Mapping):
            return self.__class__(val)
        return val

    def __setattr__(self, attr, value):
        if attr in self.dynamic_attributes:
            super().__setattr__(attr, value)
        if isinstance(value, typing.Mapping):
            self._cfg[attr].update(value)
        else:
            self._cfg[attr] = value

    def __dir__(self):
        return list(self._cfg) + dir(self.__class__)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self._cfg!r})"

    def update(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                if isinstance(value, typing.Mapping):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

    def reload(self):
        self.__init__(ConfigLoader.reload_config())

    def dump_yaml(self, stream=None):
        return yaml.dump(self._cfg, stream, default_flow_style=False)

    @classmethod
    def from_dict(cls, dict_):
        return cls(copy.deepcopy(dict_))

    def to_dict(self):
        return copy.copy(self._cfg)



