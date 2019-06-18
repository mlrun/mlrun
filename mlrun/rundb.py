import json
from os import path
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

    def store_run(self, execution, elements=[], commit=False):
        pass

    def read_run(self, uid):
        pass


class FileRunDB(RunDBInterface):

    def __init__(self, dirpath='', format='.yaml'):
        self.format = format
        if dirpath.endswith('/'):
            self.fullpath = ''
            self.dirpath = dirpath
        else:
            self.fullpath = dirpath
            self.dirpath = ''
        self._datastore = None
        self._subpath = None

    def connect(self, secrets=None):
        sm =StoreManager(secrets)
        self._datastore, self._subpath = sm.get_or_create_store(
            self.fullpath or self.dirpath)

    def store_run(self, execution, elements=[], commit=False):
        if self.format == '.yaml':
            data = execution.to_yaml()
        else:
            data = execution.to_json()
        filepath = self.fullpath or self._filepath(execution.uid, execution.project)
        self._datastore.put(filepath, data)

    def read_run(self, uid, project='default'):
        filepath = self.fullpath or self._filepath(uid, project)
        data = self._datastore.get(filepath)
        if self.format == '.yaml':
            return yaml.loads(data)
        else:
            return json.loads(data)

    def _filepath(self, uid, project):
        if project:
            return path.join(self.dirpath, '{}/{}{}'.format(project, uid, self.format))
        else:
            return path.join(self.dirpath, '{}{}'.format(uid, self.format))
