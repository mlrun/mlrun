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
from os import path
from tempfile import mktemp

import yaml

from ..datastore import StoreManager
from .base import Artifact
from ..utils import DB_SCHEMA

model_spec_filename = 'model_spec.yaml'


class ModelArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['model_file', 'metrics', 'parameters',
                                            'inputs', 'outputs', 'extra_data']
    kind = 'model'

    def __init__(self, key=None, body=None, format=None, model_file=None,
                 metrics=None, target_path=None, parameters=None,
                 inputs=None, outputs=None, extra_data=None):

        super().__init__(key, body, format=format, target_path=target_path)
        self.model_file = model_file
        self.parameters = parameters or {}
        self.metrics = metrics or {}
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.extra_data = extra_data or {}

    @property
    def is_dir(self):
        return True

    def before_log(self):
        if not self.model_file:
            raise ValueError('model_file attr must be specified')

        for key, item in self.extra_data.items():
            if hasattr(item, 'target_path'):
                self.extra_data[key] = item.target_path

    def upload(self, data_stores):

        def get_src_path(filename):
            if self.src_path:
                return path.join(self.src_path, filename)
            return filename

        target_model_path = path.join(self.target_path, self.model_file)
        body = self.get_body()
        if body:
            self._upload_body(body, data_stores, target=target_model_path)
        else:
            src_model_path = get_src_path(self.model_file)
            if not path.isfile(src_model_path):
                raise ValueError('model file {} not found'.format(src_model_path))
            self._upload_file(src_model_path, data_stores, target=target_model_path)

        spec_path = path.join(self.target_path, model_spec_filename)
        data_stores.object(url=spec_path).put(self.to_yaml())

        for key, item in self.extra_data.items():

            if isinstance(item, bytes):
                target = path.join(self.target_path, key)
                data_stores.object(url=target).put(item)
                self.extra_data[key] = target

            elif not (item.startswith('/') or '://' in item):
                src_path = get_src_path(item)
                if not path.isfile(src_path):
                    raise ValueError('extra data file {} not found'.format(src_path))
                target = path.join(self.target_path, item)
                data_stores.object(url=target).upload(src_path)


def get_model(model_dir, suffix='', stores: StoreManager = None):
    """return model file, model spec object, and list of extra data items"""
    model_file = ''
    model_spec = None
    extra_dataitems = {}
    suffix = suffix or '.pkl'
    stores = stores or StoreManager()

    if model_dir.startswith(DB_SCHEMA + '://'):
        model_spec, target = stores.get_store_artifact(model_dir)
        if not model_spec or model_spec.kind != 'model':
            raise ValueError('store artifact ({}) is not model kind'.format(model_dir))
        model_file = _get_file_path(target, model_spec.model_file)
        extra_dataitems = _get_extra(stores, target, model_spec.extra_data)

    elif model_dir.lower().endswith('.yaml'):
        model_spec = _load_model_spec(model_dir, stores)
        model_file = _get_file_path(model_dir, model_spec.model_file)
        extra_dataitems = _get_extra(stores, model_dir, model_spec.extra_data)

    elif model_dir.endswith(suffix):
        model_file = model_dir
    else:
        dirobj = stores.object(url=model_dir)
        model_dir_list = dirobj.listdir()
        if model_spec_filename in model_dir_list:
            model_spec = _load_model_spec(path.join(model_dir, model_spec_filename), stores)
            model_file = _get_file_path(model_dir, model_spec.model_file, isdir=True)
            extra_dataitems = _get_extra(stores, model_dir, model_spec.extra_data, is_dir=True)
        else:
            extra_dataitems = _get_extra(stores, model_dir,
                                         {v: v for v in model_dir_list}, is_dir=True)
            for file in model_dir_list:
                if file.endswith(suffix):
                    model_file = path.join(model_dir, file)
                    break
    if not model_file:
        raise ValueError('cant resolve model file for {} suffix{}'.format(
            model_dir, suffix))

    obj = stores.object(url=model_file)
    if obj.kind == 'file':
        return model_file, model_spec, extra_dataitems

    tmp = mktemp(suffix)
    obj.download(tmp)
    return tmp, model_spec, extra_dataitems


def _load_model_spec(specpath, stores: StoreManager):
    data = stores.object(url=specpath).get()
    spec = yaml.load(data, Loader=yaml.FullLoader)
    return ModelArtifact.from_dict(spec)


def _get_file_path(base_path: str, name: str, isdir=False):
    if name.startswith('/') or '://' in name:
        return name
    if not isdir:
        base_path = path.dirname(base_path)
    return path.join(base_path, name)


def _get_extra(stores, target, extra_data, is_dir=False):
    extra_dataitems = {}
    for k, v in extra_data.items():
        extra_dataitems[k] = stores.object(url=_get_file_path(target, v, isdir=is_dir), key=k)
    return extra_dataitems
