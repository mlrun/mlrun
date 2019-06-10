from .utils import list2dict
from os import environ

SECRET_SOURCE_KEY = 'secret_sources'


class SecretsStore:
    def __init__(self):
        self._secrets = {}

    def from_dict(self, struct: dict):
        src_list = struct.get(SECRET_SOURCE_KEY)
        if src_list and isinstance(src_list, list):
            for src in src_list:
                self._add_source(src['kind'], src.get('source'), src.get('prefix', ''))

    def to_dict(self, struct):
        pass

    def _add_source(self, kind, source={}, prefix=''):

        if kind == 'inline':
            for k, v in source.items():
                self._secrets[prefix + k] = str(v)

        elif kind == 'file':
            with open(source) as fp:
                lines = fp.read().splitlines()
                secrets_dict = list2dict(lines)
                for k, v in secrets_dict.items():
                    self._secrets[prefix + k] = str(v)

        elif kind == 'env':
            for k, v in environ.items():
                self._secrets[prefix + k] = str(v)

    def get(self, key):
        return self._secrets.get(key)
