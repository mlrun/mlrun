# Copyright 2023 Iguazio
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

import mlrun
import mlrun.artifacts
from mlrun.config import config
from mlrun.utils.helpers import parse_artifact_uri

from ..common.helpers import parse_versioned_object_uri
from ..platforms.iguazio import parse_path
from ..utils import DB_SCHEMA, StorePrefix
from .targets import get_online_target


def is_store_uri(url):
    """detect if the uri starts with the store schema prefix"""
    if not url:
        return False
    return url.startswith(DB_SCHEMA + "://")


def parse_store_uri(url):
    """parse a store uri and return kind + uri suffix"""
    if not is_store_uri(url):
        return None, ""
    uri = url[len(DB_SCHEMA) + len("://") :].strip("/")
    split = uri.split("/", 1)
    if len(split) == 0:
        raise ValueError(f"url {url} has no path")
    if split and StorePrefix.is_prefix(split[0]):
        return split[0], split[1]
    return StorePrefix.Artifact, uri


def get_store_uri(kind, uri):
    """return uri from store kind and suffix"""
    return f"{DB_SCHEMA}://{kind}/{uri}"


class ResourceCache:
    """Resource cache for real-time pipeline/serving and storey
    this cache is basic and doesnt have sync or ttl logic
    """

    def __init__(self):
        self._tabels = {}
        self._resources = {}

    def cache_table(self, uri, value, is_default=False):
        """Cache storey Table objects"""
        self._tabels[uri] = value
        if is_default:
            self._tabels["."] = value

    def get_table(self, uri):
        """get storey Table object by uri"""
        try:
            from storey import Driver, Table, V3ioDriver
        except ImportError:
            raise ImportError("storey package is not installed, use pip install storey")
        if uri in self._tabels:
            return self._tabels[uri]
        if uri in [".", ""] or uri.startswith("$"):  # $.. indicates in-mem table
            self._tabels[uri] = Table("", Driver())
            return self._tabels[uri]

        if uri.startswith("v3io://") or uri.startswith("v3ios://"):
            endpoint, uri = parse_path(uri)
            self._tabels[uri] = Table(
                uri,
                V3ioDriver(webapi=endpoint or mlrun.mlconf.v3io_api),
                flush_interval_secs=mlrun.mlconf.feature_store.flush_interval,
            )
            return self._tabels[uri]

        if uri.startswith("redis://") or uri.startswith("rediss://"):
            from storey.redis_driver import RedisDriver

            endpoint, uri = parse_path(uri)
            endpoint = endpoint or mlrun.mlconf.redis.url
            self._tabels[uri] = Table(
                uri,
                RedisDriver(redis_url=endpoint, key_prefix="/"),
                flush_interval_secs=mlrun.mlconf.feature_store.flush_interval,
            )
            return self._tabels[uri]

        if is_store_uri(uri):
            resource = get_store_resource(uri)
            if resource.kind in [
                mlrun.common.schemas.ObjectKind.feature_set.value,
                mlrun.common.schemas.ObjectKind.feature_vector.value,
            ]:
                target = get_online_target(resource)
                if not target:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"resource {uri} does not have an online data target"
                    )
                self._tabels[uri] = target.get_table_object()
                return self._tabels[uri]

        raise mlrun.errors.MLRunInvalidArgumentError(f"table {uri} not found in cache")

    def cache_resource(self, uri, value, default=False):
        """cache store resource (artifact/feature-set/feature-vector)"""
        self._resources[uri] = value
        if default:
            self._resources["."] = value

    def get_resource(self, uri):
        """get resource from cache by uri"""
        return self._resources[uri]

    def resource_getter(self, db=None, secrets=None):
        """wraps get_store_resource with a simple object cache"""

        def _get_store_resource(uri, use_cache=True):
            """get mlrun store resource object
            :param use_cache: indicate if we read from local cache or from DB
            """
            if (uri == "." or use_cache) and uri in self._resources:
                return self._resources[uri]
            resource = get_store_resource(uri, db, secrets=secrets)
            if use_cache:
                self._resources[uri] = resource
            return resource

        return _get_store_resource


def get_store_resource(
    uri, db=None, secrets=None, project=None, data_store_secrets=None
):
    """get store resource object by uri"""

    db = db or mlrun.get_run_db(secrets=secrets)
    kind, uri = parse_store_uri(uri)
    if not kind:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Cannot get store resource from invalid URI: {uri}"
        )
    elif kind == StorePrefix.FeatureSet:
        project, name, tag, uid = parse_versioned_object_uri(
            uri, project or config.default_project
        )
        return db.get_feature_set(name, project, tag, uid)

    elif kind == StorePrefix.FeatureVector:
        project, name, tag, uid = parse_versioned_object_uri(
            uri, project or config.default_project
        )
        return db.get_feature_vector(name, project, tag, uid)

    elif StorePrefix.is_artifact(kind):
        project, key, iteration, tag, tree, uid = parse_artifact_uri(
            uri, project or config.default_project
        )
        resource = db.read_artifact(
            key,
            project=project,
            tag=tag,
            iter=iteration,
            tree=tree,
            uid=uid,
        )
        if resource.get("kind", "") == "link":
            # todo: support other link types (not just iter, move this to the db/api layer
            link_iteration = resource["spec"].get("link_iteration", 0)

            resource = db.read_artifact(
                key,
                tag=tag,
                iter=link_iteration,
                project=project,
            )
        if resource:
            return mlrun.artifacts.dict_to_artifact(resource)

    else:
        stores = mlrun.store_manager.set(secrets, db=db)
        return stores.object(url=uri, secrets=data_store_secrets)
