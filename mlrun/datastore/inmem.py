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

from io import BytesIO, StringIO

import pandas as pd

import mlrun

from .base import DataStore, FileStats


class InMemoryStore(DataStore):
    def __init__(self):
        super().__init__(None, "memory", "memory", "")
        self._items = {}

    @property
    def url(self):
        return "memory://"

    def _get_parent_secret(self, key):
        return None

    def _get_item(self, key):
        if key not in self._items:
            raise ValueError(f"item {key} not found in memory store")
        return self._items[key]

    def get(self, key, size=None, offset=0):
        item = self._get_item(key)
        return item

    def put(self, key, data, append=False):
        if append and key in self._items:
            self._items[key] = self._items[key] + data
        else:
            self._items[key] = data

    def upload(self, key, src_path):
        with open(src_path, "rb") as fp:
            self._items[key] = fp.read()

    def stat(self, key):
        return FileStats(size=len(self._get_item(key)), modified=0)

    def listdir(self, key):
        return []

    def as_df(self, url, subpath, columns=None, df_module=None, format="", **kwargs):
        item = self._get_item(subpath)
        if hasattr(item, "to_csv"):  # detect if it is a dataframe type
            return item
        if isinstance(item, str):
            item = StringIO(item)
        else:
            item = BytesIO(item)

        df_module = df_module or pd
        if url.endswith(".csv") or format == "csv":
            if columns:
                kwargs["usecols"] = columns
            reader = df_module.read_csv
        elif url.endswith(".parquet") or url.endswith(".pq") or format == "parquet":
            if columns:
                kwargs["columns"] = columns
            reader = df_module.read_parquet
        elif url.endswith(".json") or format == "json":
            reader = df_module.read_json
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"file type unhandled {url}")

        return reader(item, **kwargs)
