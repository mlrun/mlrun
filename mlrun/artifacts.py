import json
import os

import yaml

from .datastore import StoreManager
from .rundb import RunDBInterface
from .utils import uxjoin, run_keys, ModelObj




class ArtifactManager:

    def __init__(self, stores: StoreManager,execution=None,
                 db: RunDBInterface = None,
                 out_path='',
                 calc_hash=True):
        self._execution = execution
        self.out_path = out_path
        self.calc_hash = calc_hash

        self.data_stores = stores
        self.artifact_db = db
        self.input_artifacts = {}
        self.output_artifacts = {}
        self.outputs_spec = {}

    def from_dict(self, struct: dict):
        self.out_path = struct.get('default_output_path', self.out_path)
        out_list = struct.get(run_keys.output_artifacts)
        if out_list and isinstance(out_list, list):
            for item in out_list:
                self.outputs_spec[item['key']] = item.get('path')

    def to_dict(self, struct):
        struct['spec'][run_keys.output_artifacts] = [{'key':k, 'path':v} for k, v in self.outputs_spec.items()]
        struct['spec'][run_keys.output_path] = self.out_path
        struct['status'][run_keys.output_artifacts] = [item.base_dict() for item in self.output_artifacts.values()]

    def log_artifact(self, item, body=None, src_path='', target_path='', tag=''):
        if isinstance(item, str):
            key = item
            item = Artifact(key, body)
        else:
            key = item.key
            target_path = target_path or item.target_path

        if key in self.outputs_spec.keys():
            target_path = self.outputs_spec[key] or target_path
        if not target_path:
            target_path = uxjoin(self.out_path, key)
        item.target_path = target_path
        item.tag = tag or item.tag or self._execution.tag
        item.src_path = src_path

        self.output_artifacts[key] = item
        store, ipath = self.get_store(target_path)

        body = body or item.get_body()

        if body:
            store.put(ipath, body)
        else:
            src_path = src_path or key
            if os.path.isfile(src_path):
                store.upload(ipath, src_path)

        if self.artifact_db:
            if not item.sources:
                item.sources = self._execution.to_dict()['spec'][run_keys.input_objects]
            item.execution = self._execution.get_meta()
            self.artifact_db.store_artifact(key, item, item.tag, self._execution.project)

    def get_store(self, url):
        return self.data_stores.get_or_create_store(url)


class Artifact(ModelObj):
    _dict_fields = ['key', 'src_path', 'target_path', 'hash', 'description']

    def __init__(self, key, body=None, src_path='', target_path='', tag=''):
        self.kind = ''
        self._key = key
        self.tag = tag
        self.target_path = target_path
        self.src_path = src_path
        self._body = body
        self.description = ''
        self.format = ''
        self.encoding = ''
        self.sources = []
        self.execution = None
        self.hash = None
        self.license = ''

    @property
    def key(self):
        return self._key

    def get_body(self):
        return self._body

    def base_dict(self):
        return super().to_dict()

    def to_dict(self):
        return super().to_dict(self._dict_fields + ['execution', 'sources'])


class Table(Artifact):
    def __init__(self):
        super().__init__()
        self.kind = 'table'
        self.schema = None
