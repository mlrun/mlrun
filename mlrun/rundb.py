import json
from os import path, environ
from urllib.parse import urlparse
import yaml
from .datastore import StoreManager


def get_run_db(url=''):
    p = urlparse(url)
    scheme = p.scheme.lower()
    if '://' not in url or scheme in ['file', 's3', 'v3io', 'v3ios']:
        db = FileRunDB(url)
    else:
        raise ValueError('unsupported run DB scheme ({})'.format(scheme))
    return db


class RunDBInterface:

    def connect(self, secrets=None):
        pass

    def store_run(self, execution, elements=[], commit=False):
        pass

    def read_run(self, uid):
        pass

    def store_artifact(self, key, artifact, tag='', project=''):
        pass

    def read_artifact(self, key, tag='', project=''):
        pass

    def store_metric(self, keyvals={}, timestamp=None, labels={}):
        pass

    def read_metric(self, keys, query=''):
        pass


class FileRunDB(RunDBInterface):

    def __init__(self, dirpath='', format='.yaml'):
        self.format = format
        self.dirpath = dirpath
        self._datastore = None
        self._subpath = None

    def connect(self, secrets=None):
        sm =StoreManager(secrets)
        self._datastore, self._subpath = sm.get_or_create_store(self.dirpath)

    def store_run(self, execution, elements=[], commit=False):
        if self.format == '.yaml':
            data = execution.to_yaml()
        else:
            data = execution.to_json()
        filepath = self._filepath(execution.uid, '', execution.project, 'runs')
        self._datastore.put(filepath, data)

    def read_run(self, uid, project='default'):
        filepath = self._filepath(uid, '', project, 'runs')
        data = self._datastore.get(filepath)
        if self.format == '.yaml':
            return yaml.load(data)
        else:
            return json.loads(data)

    def store_artifact(self, key, artifact, tag='', project=''):
        if self.format == '.yaml':
            data = artifact.to_yaml()
        else:
            data = artifact.to_json()
        filepath = self._filepath(key, tag, project, 'artifacts')
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag='', project=''):
        filepath = self._filepath(key, tag, project, 'artifacts')
        data = self._datastore.get(filepath)
        if self.format == '.yaml':
            return yaml.load(data)
        else:
            return json.loads(data)

    def _filepath(self, uid, tag, project, table):
        if tag:
            tag = '/' + tag
        if project:
            return path.join(self.dirpath, '{}/{}/{}{}{}'.format(table, project, uid, tag, self.format))
        else:
            return path.join(self.dirpath, '{}/{}{}{}'.format(table, uid, tag, self.format))
