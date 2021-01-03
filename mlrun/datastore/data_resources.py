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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import mlrun
from mlrun.config import config
from mlrun.utils.helpers import parse_function_uri
from mlrun.artifacts import dict_to_artifact
from .v3io import v3io_path
from ..utils import StorePrefix, parse_store_uri


class ResourceCache:
    def __init__(self):
        self._tabels = {}
        self._resources = {}

    def cache_table(self, uri, value, is_default=False):
        self._tabels[uri] = value
        if is_default:
            self._tabels["."] = value

    def get_table(self, uri):
        try:
            from storey import Table, Driver, V3ioDriver
        except ImportError:
            raise ImportError("storey package is not installed, use pip install storey")

        if uri in self._tabels:
            return self._tabels[uri]
        if uri in [".", ""]:
            self._tabels[uri] = Table("", Driver())
            return self._tabels[uri]

        if uri.startswith("v3io://") or uri.startswith("v3ios://"):
            endpoint, uri = v3io_path(uri)
            self._tabels[uri] = Table(uri, V3ioDriver(webapi=endpoint))
            return self._tabels[uri]

        # todo: map store:// uri's to Table objects

        raise ValueError(f"table {uri} not found in cache")

    def cache_resource(self, uri, value, default=False):
        self._resources[uri] = value
        if default:
            self._resources["."] = value

    def get_resource(self, uri):
        return self._resources[uri]

    def resource_getter(self, db=None, secrets=None):
        """wraps get_data_resource with a simple object cache"""

        def _get_data_resource(uri, use_cache=True):
            if (uri == "." or use_cache) and uri in self._resources:
                return self._resources[uri]
            resource = get_data_resource(uri, db, secrets=secrets)
            if use_cache:
                self._resources[uri] = resource
            return resource

        return _get_data_resource


def get_data_resource(uri, db=None, secrets=None):
    db = db or mlrun.get_run_db(secrets=secrets)
    kind, uri = parse_store_uri(uri)
    if kind == StorePrefix.FeatureSet:
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        return db.get_feature_set(name, project, tag, uid)

    elif kind == StorePrefix.FeatureVector:
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        return db.get_feature_vector(name, project, tag, uid)

    elif StorePrefix.is_artifact(kind):
        project, name, tag, uid = parse_function_uri(uri, config.default_project)
        resp = db.read_artifact(name, project=project, tag=tag or uid)
        if resp:
            return dict_to_artifact(resp)

    else:
        stores = mlrun.store_manager.set(secrets, db=db)
        return stores.object(url=uri)
