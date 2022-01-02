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
import sys
import tempfile
from base64 import b64encode
from os import getenv, path, remove

import dask.dataframe as dd
import fsspec
import orjson
import pandas as pd
import requests
import urllib3

import mlrun.errors
from mlrun.secrets import SecretsStore
from mlrun.utils import is_ipython, logger

verify_ssl = False
if not verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileStats:
    def __init__(self, size, modified, content_type=None):
        self.size = size
        self.modified = modified
        self.content_type = content_type

    def __repr__(self):
        return f"FileStats(size={self.size}, modified={self.modified}, type={self.content_type})"


class DataStore:
    def __init__(self, parent, name, kind, endpoint=""):
        self._parent = parent
        self.kind = kind
        self.name = name
        self.endpoint = endpoint
        self.subpath = ""
        self.secret_pfx = ""
        self.options = {}
        self.from_spec = False
        self._filesystem = None

    @property
    def is_structured(self):
        return False

    @property
    def is_unstructured(self):
        return True

    @staticmethod
    def uri_to_kfp(endpoint, subpath):
        raise ValueError("data store doesnt support KFP URLs")

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return ""

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        return None

    def supports_isdir(self):
        """Whether the data store supports isdir"""
        return True

    def _get_secret_or_env(self, key, default=None):
        # Project-secrets are mounted as env variables whose name can be retrieved from SecretsStore
        return (
            self._secret(key)
            or getenv(key)
            or getenv(SecretsStore.k8s_env_variable_name_for_secret(key))
            or default
        )

    def get_storage_options(self):
        """get fsspec storage options"""
        return None

    def open(self, filepath, mode):
        fs = self.get_filesystem(False)
        return fs.open(filepath, mode)

    def _join(self, key):
        if self.subpath:
            return f"{self.subpath}/{key}"
        return key

    def _secret(self, key):
        return self._parent.secret(self.secret_pfx + key)

    @property
    def url(self):
        return f"{self.kind}://{self.endpoint}"

    def get(self, key, size=None, offset=0):
        pass

    def query(self, key, query="", **kwargs):
        raise ValueError("data store doesnt support structured queries")

    def put(self, key, data, append=False):
        pass

    def stat(self, key):
        pass

    def listdir(self, key):
        raise ValueError("data store doesnt support listdir")

    def download(self, key, target_path):
        data = self.get(key)
        mode = "wb"
        if isinstance(data, str):
            mode = "w"
        with open(target_path, mode) as fp:
            fp.write(data)
            fp.close()

    def upload(self, key, src_path):
        pass

    def as_df(
        self,
        url,
        subpath,
        columns=None,
        df_module=None,
        format="",
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        df_module = df_module or pd
        if url.endswith(".csv") or format == "csv":
            if columns:
                kwargs["usecols"] = columns
            reader = df_module.read_csv
        elif url.endswith(".parquet") or url.endswith(".pq") or format == "parquet":
            if columns:
                kwargs["columns"] = columns

            def reader(*args, **kwargs):
                if start_time or end_time:
                    if sys.version_info < (3, 7):
                        raise ValueError(
                            f"feature not supported for python version {sys.version_info}"
                        )

                    from storey.utils import find_filters, find_partitions

                    filters = []
                    partitions_time_attributes = find_partitions(url, fs)

                    find_filters(
                        partitions_time_attributes,
                        start_time,
                        end_time,
                        filters,
                        time_column,
                    )
                    kwargs["filters"] = filters

                return df_module.read_parquet(*args, **kwargs)

        elif url.endswith(".json") or format == "json":
            reader = df_module.read_json

        else:
            raise Exception(f"file type unhandled {url}")

        fs = self.get_filesystem()
        if fs:
            if self.supports_isdir() and fs.isdir(url) or df_module == dd:
                storage_options = self.get_storage_options()
                if storage_options:
                    kwargs["storage_options"] = storage_options
                return reader(url, **kwargs)
            else:

                file = url
                # Workaround for ARROW-12472 affecting pyarrow 3.x and 4.x.
                if fs.protocol != "file":
                    # If not dir, use fs.open() to avoid regression when pandas < 1.2 and does not
                    # support the storage_options parameter.
                    file = fs.open(url)

                return reader(file, **kwargs)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.download(self._join(subpath), temp_file.name)
        df = reader(temp_file.name, **kwargs)
        remove(temp_file.name)
        return df

    def to_dict(self):
        return {
            "name": self.name,
            "url": f"{self.kind}://{self.endpoint}/{self.subpath}",
            "secret_pfx": self.secret_pfx,
            "options": self.options,
        }

    def rm(self, path, recursive=False, maxdepth=None):
        self.get_filesystem().rm(path=path, recursive=recursive, maxdepth=maxdepth)


class DataItem:
    """Data input/output class abstracting access to various local/remote data sources

    DataItem objects are passed into functions and can be used inside the function, when a function run completes
    users can access the run data via the run.artifact(key) which returns a DataItem object.
    users can also convert a data url (e.g. s3://bucket/key.csv) to a DataItem using `mlrun.get_dataitem(url)`.

    Example::

        # using data item inside a function
        def my_func(context, data: DataItem):
            df = data.as_df()


        # reading run results using DataItem (run.artifact())
        train_run = train_iris_func.run(inputs={'dataset': dataset},
                                        params={'label_column': 'label'})

        train_run.artifact('confusion-matrix').show()
        test_set = train_run.artifact('test_set').as_df()

        # create and use DataItem from uri
        data = mlrun.get_dataitem('http://xyz/data.json').get()
    """

    def __init__(
        self,
        key: str,
        store: DataStore,
        subpath: str,
        url: str = "",
        meta=None,
        artifact_url=None,
    ):
        self._store = store
        self._key = key
        self._url = url
        self._path = subpath
        self._meta = meta
        self._artifact_url = artifact_url
        self._local_path = ""

    @property
    def key(self):
        """DataItem key"""
        return self._key

    @property
    def suffix(self):
        """DataItem suffix (file extension) e.g. '.png'"""
        _, file_ext = path.splitext(self._path)
        return file_ext

    @property
    def store(self):
        """DataItem store object"""
        return self._store

    @property
    def kind(self):
        """DataItem store kind (file, s3, v3io, ..)"""
        return self._store.kind

    @property
    def meta(self):
        """Artifact Metadata, when the DataItem is read from the artifacts store"""
        return self._meta

    @property
    def artifact_url(self):
        """DataItem artifact url (when its an artifact) or url for simple dataitems"""
        return self._artifact_url or self._url

    @property
    def url(self):
        """DataItem url e.g. /dir/path, s3://bucket/path"""
        return self._url

    def get(self, size=None, offset=0, encoding=None):
        """read all or a byte range and return the content

        :param size:     number of bytes to get
        :param offset:   fetch from offset (in bytes)
        :param encoding: encoding (e.g. "utf-8") for converting bytes to str
        """
        body = self._store.get(self._path, size=size, offset=offset)
        if encoding and isinstance(body, bytes):
            body = body.decode(encoding)
        return body

    def download(self, target_path):
        """download to the target dir/path

        :param target_path: local target path for the downloaded item
        """
        self._store.download(self._path, target_path)

    def put(self, data, append=False):
        """write/upload the data, append is only supported by some datastores

        :param data:   data (bytes/str) to write
        :param append: append data to the end of the object, NOT SUPPORTED BY SOME OBJECT STORES!
        """
        self._store.put(self._path, data, append=append)

    def upload(self, src_path):
        """upload the source file (src_path)

        :param src_path: source file path to read from and upload
        """
        self._store.upload(self._path, src_path)

    def stat(self):
        """return FileStats class (size, modified, content_type)"""
        return self._store.stat(self._path)

    def open(self, mode):
        """return fsspec file handler, if supported"""
        return self._store.open(self._url, mode)

    def ls(self):
        """return a list of child file names"""
        return self._store.listdir(self._path)

    def listdir(self):
        """return a list of child file names"""
        return self._store.listdir(self._path)

    def local(self):
        """get the local path of the file, download to tmp first if its a remote object"""
        if self.kind == "file":
            return self._path
        if self._local_path:
            return self._local_path

        dot = self._path.rfind(".")
        suffix = "" if dot == -1 else self._path[dot:]
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        self._local_path = temp_file.name
        logger.info(f"downloading {self.url} to local temp file")
        self.download(self._local_path)
        return self._local_path

    def as_df(
        self, columns=None, df_module=None, format="", **kwargs,
    ):
        """return a dataframe object (generated from the dataitem).

        :param columns:   optional, list of columns to select
        :param df_module: optional, py module used to create the DataFrame (e.g. pd, dd, cudf, ..)
        :param format:    file format, if not specified it will be deducted from the suffix
        """
        return self._store.as_df(
            self._url,
            self._path,
            columns=columns,
            df_module=df_module,
            format=format,
            **kwargs,
        )

    def show(self, format=None):
        """show the data object content in Jupyter

        :param format: format to use (when there is no/wrong suffix), e.g. 'png'
        """
        if not is_ipython:
            logger.warning(
                "Jupyter/IPython was not detected, .show() will only display inside Jupyter"
            )
            return

        from IPython import display

        suffix = self.suffix.lower()
        if format:
            suffix = "." + format

        if suffix in [".jpg", ".png", ".gif"]:
            display.display(display.Image(self.get(), format=suffix[1:]))
        elif suffix in [".htm", ".html"]:
            display.display(display.HTML(self.get(encoding="utf-8")))
        elif suffix in [".csv", ".pq", ".parquet"]:
            display.display(self.as_df())
        elif suffix in [".yaml", ".txt", ".py"]:
            display.display(display.Pretty(self.get(encoding="utf-8")))
        elif suffix == ".json":
            display.display(display.JSON(orjson.loads(self.get())))
        elif suffix == ".md":
            display.display(display.Markdown(self.get(encoding="utf-8")))
        else:
            logger.error(f"unsupported show() format {suffix} for {self.url}")

    def __str__(self):
        return self.url

    def __repr__(self):
        return f"'{self.url}'"


def get_range(size, offset):
    byterange = f"bytes={offset}-"
    if size:
        byterange += str(offset + size)
    return byterange


def basic_auth_header(user, password):
    username = user.encode("latin1")
    password = password.encode("latin1")
    base = b64encode(b":".join((username, password))).strip()
    authstr = "Basic " + base.decode("ascii")
    return {"Authorization": authstr}


def http_get(url, headers=None, auth=None):
    try:
        response = requests.get(url, headers=headers, auth=auth, verify=verify_ssl)
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {exc}")

    mlrun.errors.raise_for_status(response)

    return response.content


def http_head(url, headers=None, auth=None):
    try:
        response = requests.head(url, headers=headers, auth=auth, verify=verify_ssl)
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {exc}")

    mlrun.errors.raise_for_status(response)

    return response.headers


def http_put(url, data, headers=None, auth=None):
    try:
        response = requests.put(
            url, data=data, headers=headers, auth=auth, verify=verify_ssl
        )
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {exc}")

    mlrun.errors.raise_for_status(response)


def http_upload(url, file_path, headers=None, auth=None):
    with open(file_path, "rb") as data:
        http_put(url, data, headers, auth)


class HttpStore(DataStore):
    def __init__(self, parent, schema, name, endpoint=""):
        super().__init__(parent, name, schema, endpoint)
        self.auth = None

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if not self._filesystem:
            self._filesystem = fsspec.filesystem("http")
        return self._filesystem

    def supports_isdir(self):
        return False

    def upload(self, key, src_path):
        raise ValueError("unimplemented")

    def put(self, key, data, append=False):
        raise ValueError("unimplemented")

    def get(self, key, size=None, offset=0):
        data = http_get(self.url + self._join(key), None, self.auth)
        if offset:
            data = data[offset:]
        if size:
            data = data[:size]
        return data
