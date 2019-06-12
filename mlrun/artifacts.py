from os import path
from mlrun.datastore import StoreManager

INPUT_ARTIFACT_KEY = 'input_artifacts'
OUTPUT_ARTIFACT_KEY = 'output_artifacts'


class ArtifactManager:

    def __init__(self, stores: StoreManager,
                 getmeta, in_path='', out_path=''):
        self._getmeta_function = getmeta
        self.in_path = in_path
        self.out_path = out_path

        self.data_stores = stores
        self.input_artifacts = {}
        self.output_artifacts = {}
        self.outputs_spec = {}

    def from_dict(self, struct: dict):
        self.in_path = struct.get('default_input_path', self.in_path)
        self.out_path = struct.get('default_output_path', self.out_path)
        in_list = struct.get(INPUT_ARTIFACT_KEY)
        if in_list and isinstance(in_list, list):
            for item in in_list:
                artifact = InputArtifact(self, item['key'], item.get('path'))
                self.input_artifacts[item['key']] = artifact

        out_list = struct.get(OUTPUT_ARTIFACT_KEY)
        if out_list and isinstance(out_list, list):
            for item in out_list:
                self.outputs_spec[item['key']] = item.get('path')

    def to_dict(self, struct):
        struct['spec'][INPUT_ARTIFACT_KEY] = [item.to_dict() for item in self.input_artifacts.values()]
        struct['spec'][OUTPUT_ARTIFACT_KEY] = [{'key':k, 'path':v} for k, v in self.outputs_spec.items()]
        struct['spec']['default_input_path'] = self.in_path
        struct['spec']['default_output_path'] = self.out_path
        struct['status'][OUTPUT_ARTIFACT_KEY] = [item.to_dict() for item in self.output_artifacts.values()]

    def get_input_artifact(self, key):
        if key not in self.input_artifacts:
            artifact = InputArtifact(self, key)
            self.input_artifacts[key] = artifact
        else:
            artifact = self.input_artifacts[key]
        return artifact

    def log_artifact(self, key, target_path='', body=None, atype='', source_path=''):
        if key in self.outputs_spec:
            target_path = self.outputs_spec[key]
        artifact = OutputArtifact(self, key, target_path, atype, source_path)
        self.output_artifacts[key] = artifact

        if body:
            artifact.put(body)
        else:
            artifact.upload(artifact.source_path)

    def get_store(self, url):
        return self.data_stores.get_or_create_store(url)


class DataArtifact:

    def __init__(self, parent: ArtifactManager, key, realpath=''):
        self._parent = parent
        self._key = key
        self._realpath = realpath
        self._store = None
        self._path = ''

    @property
    def url(self):
        return self._realpath or self._key

    def _init_store(self):
        self._store, self._path = self._parent.get_store(self._realpath)

    def get(self):
        if self._store:
            return self._store.get(self._path)

    def download(self, target_path):
        if self._store:
            self._store.download(self._path, target_path)

    def put(self, data):
        if self._store:
            self._store.put(self._path, data)

    def upload(self, src_path):
        if self._store:
            self._store.upload(self._path, src_path)

    def to_dict(self):
        return {
            'key': self._key,
            'path': self._realpath,
        }


class InputArtifact(DataArtifact):

    def __init__(self, parent: ArtifactManager, key, realpath=''):
        if not realpath:
            realpath = path.join(parent.in_path, key)
        super().__init__(parent, key, realpath)
        self._init_store()


class OutputArtifact(DataArtifact):

    def __init__(self, parent, key, target_path='', atype='', source_path=''):
        if not target_path:
            target_path = path.join(parent.out_path, key)
        super().__init__(parent, key, target_path)
        self._init_store()
        self.atype = atype
        self.source_path = source_path or key

    def to_dict(self):
        return {
            'key': self._key,
            'target_path': self._realpath,
            'atype': self.atype,
            'source_path': self.source_path,
        }
