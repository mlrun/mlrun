from .utils import list2dict
from os import environ


class SecretsStore:

    def __init__(self):
        self._secrets = {}

    def add_source(self, kind, source={}, prefix=''):

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
