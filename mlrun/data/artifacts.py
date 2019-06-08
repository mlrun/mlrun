from os import path
from .datastore import StoreManager


class ArtifactManager:

    def __init__(self, stores: StoreManager,
                 runmeta, secrets=None, in_path='', out_path=''):
        self.runmeta = runmeta
        self.secrets = secrets
        self.in_path = in_path
        self.out_path = out_path

        self.data_stores = stores
        self.input_artifact = {}
        self.output_artifact = {}

    def get_store(self, url):
        return self.data_stores.get_or_create_store(url)


class DataArtifact:

    def __init__(self, parent: ArtifactManager, key, realpath=''):
        self._parent = parent
        self._key = key
        self._realpath = realpath
        self._store, self._path = parent.get_store(realpath)

    def get(self):
        self._store.get(self._path)

    def download(self, target_path):
        self._store.download(self._path, target_path)

    def put(self, data):
        self._store.put(self._path, data)

    def upload(self, src_path):
        self._store.upload(self._path, src_path)

    def to_dict(self):
        return {
            'key': self._key,
            'path': self._realpath,
        }


class InputArtifact(DataArtifact):

    def __init__(self):
        pass


class OutputArtifact(DataArtifact):

    def __init__(self, parent, key, target_path='', body=None, atype=''):
        if not target_path:
            target_path = path.join(parent.out_path, key)
        super().__init__(parent, key, target_path)
        self.atype = atype

    def to_dict(self):
        return {
            'localpath': self._key,
            'target_path': self._realpath,
            'atype': self.atype,
        }
