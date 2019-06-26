import json

import yaml

from .datastore import StoreManager
from .rundb import RunDBInterface
from .utils import uxjoin, run_keys




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
        struct['status'][run_keys.output_artifacts] = [item.to_dict() for item in self.output_artifacts.values()]

    def log_artifact(self, item, body=None, target_path='', tag=''):
        if isinstance(item, str):
            key = item
            item = Artifact(key, body)
        else:
            key = item.key
            target_path = target_path or item.target_path

        if key in self.outputs_spec.keys():
            target_path = self.outputs_spec[key]
        if not target_path:
            target_path = uxjoin(self.out_path, key)
        item.target_path = target_path

        self.output_artifacts[key] = item
        store, ipath = self.get_store(target_path)

        body = body or item.get_body()

        if body:
            store.put(ipath, body)
        else:
            store.upload(ipath, key)

        if self.artifact_db:
            tag = tag or self._execution.tag
            if not item.sources:
                item.sources = self._execution.to_dict()['spec'][run_keys.input_objects]
            item.execution = self._execution.get_meta()
            self.artifact_db.store_artifact(key, item, tag, self._execution.project)

    def get_store(self, url):
        return self.data_stores.get_or_create_store(url)


class Artifact:

    def __init__(self, key, body=None, target_path=''):
        self.kind = ''
        self._key = key
        self.target_path = target_path
        self._store = None
        self._path = ''
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

    def to_dict(self):
        return {
            'key': self._key,
            'path': self.target_path,
            'hash': self.hash,
            'description': self.description,
            'execution': self.execution,
            'sources': self.sources,
        }

    def to_yaml(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self):
        return json.dumps(self.to_dict())




class Table(Artifact):
    def __init__(self):
        super().__init__()
        self.kind = 'table'
        self.schema = None
