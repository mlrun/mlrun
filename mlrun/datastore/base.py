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
import tempfile
import urllib.parse
from base64 import b64encode
from os import path, remove
from typing import Optional, Union

import dask.dataframe as dd
import fsspec
import orjson
import pandas as pd
import pyarrow
import pytz
import requests
import urllib3

import mlrun.errors
from mlrun.errors import err_to_str
from mlrun.utils import StorePrefix, is_ipython, logger

from .store_resources import is_store_uri, parse_store_uri
from .utils import filter_df_start_end_time, select_columns_from_df

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
    def __init__(self, parent, name, kind, endpoint="", secrets: dict = None):
        self._parent = parent
        self.kind = kind
        self.name = name
        self.endpoint = endpoint
        self.subpath = ""
        self.secret_pfx = ""
        self.options = {}
        self.from_spec = False
        self._filesystem = None
        self._secrets = secrets or {}

    @property
    def is_structured(self):
        return False

    @property
    def is_unstructured(self):
        return True

    @staticmethod
    def _sanitize_url(url):
        """
        Extract only the schema, netloc, and path from an input URL if they exist,
        excluding parameters, query, or fragments.
        """
        parsed_url = urllib.parse.urlparse(url)
        scheme = f"{parsed_url.scheme}:" if parsed_url.scheme else ""
        netloc = f"//{parsed_url.netloc}" if parsed_url.netloc else "//"
        return f"{scheme}{netloc}{parsed_url.path}"

    @staticmethod
    def uri_to_kfp(endpoint, subpath):
        raise ValueError("data store doesnt support KFP URLs")

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return ""

    def get_filesystem(self, silent=True) -> Optional[fsspec.AbstractFileSystem]:
        """return fsspec file system object, if supported"""
        return None

    def supports_isdir(self):
        """Whether the data store supports isdir"""
        return True

    def _get_secret_or_env(self, key, default=None):
        # Project-secrets are mounted as env variables whose name can be retrieved from SecretsStore
        return mlrun.get_secret_or_env(
            key, secret_provider=self._get_secret, default=default
        )

    def get_storage_options(self):
        """get fsspec storage options"""
        return None

    def open(self, filepath, mode):
        file_system = self.get_filesystem(False)
        return file_system.open(filepath, mode)

    def _join(self, key):
        if self.subpath:
            return f"{self.subpath}/{key}"
        return key

    def _get_parent_secret(self, key):
        return self._parent.secret(self.secret_pfx + key)

    def _get_secret(self, key: str, default=None):
        return self._secrets.get(key, default) or self._get_parent_secret(key)

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

    @staticmethod
    def _parquet_reader(df_module, url, file_system, time_column, start_time, end_time):
        from storey.utils import find_filters, find_partitions

        def set_filters(
            partitions_time_attributes, start_time_inner, end_time_inner, kwargs
        ):
            filters = []
            find_filters(
                partitions_time_attributes,
                start_time_inner,
                end_time_inner,
                filters,
                time_column,
            )
            kwargs["filters"] = filters

        def reader(*args, **kwargs):
            if start_time or end_time:
                if time_column is None:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "When providing start_time or end_time, must provide time_column"
                    )

                partitions_time_attributes = find_partitions(url, file_system)
                set_filters(
                    partitions_time_attributes,
                    start_time,
                    end_time,
                    kwargs,
                )
                try:
                    return df_module.read_parquet(*args, **kwargs)
                except pyarrow.lib.ArrowInvalid as ex:
                    if not str(ex).startswith(
                        "Cannot compare timestamp with timezone to timestamp without timezone"
                    ):
                        raise ex

                    if start_time.tzinfo:
                        start_time_inner = start_time.replace(tzinfo=None)
                        end_time_inner = end_time.replace(tzinfo=None)
                    else:
                        start_time_inner = start_time.replace(tzinfo=pytz.utc)
                        end_time_inner = end_time.replace(tzinfo=pytz.utc)

                    set_filters(
                        partitions_time_attributes,
                        start_time_inner,
                        end_time_inner,
                        kwargs,
                    )
                    return df_module.read_parquet(*args, **kwargs)
            else:
                return df_module.read_parquet(*args, **kwargs)

        return reader

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
        file_url = self._sanitize_url(url)
        is_csv, is_json, drop_time_column = False, False, False
        file_system = self.get_filesystem()
        if file_url.endswith(".csv") or format == "csv":
            is_csv = True
            drop_time_column = False
            if columns:
                if (
                    time_column
                    and (start_time or end_time)
                    and time_column not in columns
                ):
                    columns.append(time_column)
                    drop_time_column = True
                kwargs["usecols"] = columns

            reader = df_module.read_csv
            if file_system:
                if file_system.isdir(file_url):

                    def reader(*args, **kwargs):
                        base_path = args[0]
                        file_entries = file_system.listdir(base_path)
                        filenames = []
                        for file_entry in file_entries:
                            if (
                                file_entry["name"].endswith(".csv")
                                and file_entry["size"] > 0
                                and file_entry["type"] == "file"
                            ):
                                filename = file_entry["name"]
                                filename = filename.split("/")[-1]
                                filenames.append(filename)
                        dfs = []
                        for filename in filenames:
                            updated_args = [f"{base_path}/{filename}"]
                            updated_args.extend(args[1:])
                            dfs.append(df_module.read_csv(*updated_args, **kwargs))
                        return df_module.concat(dfs)

        elif (
            file_url.endswith(".parquet")
            or file_url.endswith(".pq")
            or format == "parquet"
        ):
            if columns:
                kwargs["columns"] = columns

            reader = self._parquet_reader(
                df_module, url, file_system, time_column, start_time, end_time
            )

        elif file_url.endswith(".json") or format == "json":
            is_json = True
            reader = df_module.read_json

        else:
            raise Exception(f"file type unhandled {url}")

        if file_system:
            if self.supports_isdir() and file_system.isdir(file_url) or df_module == dd:
                storage_options = self.get_storage_options()
                if url.startswith("ds://"):
                    parsed_url = urllib.parse.urlparse(url)
                    url = parsed_url.path[1:]
                    # Pass the underlying file system
                    kwargs["filesystem"] = file_system
                elif storage_options:
                    kwargs["storage_options"] = storage_options
                df = reader(url, **kwargs)
            else:

                file = url
                # Workaround for ARROW-12472 affecting pyarrow 3.x and 4.x.
                if file_system.protocol != "file":
                    # If not dir, use file_system.open() to avoid regression when pandas < 1.2 and does not
                    # support the storage_options parameter.
                    file = file_system.open(url)

                df = reader(file, **kwargs)
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.download(self._join(subpath), temp_file.name)
            df = reader(temp_file.name, **kwargs)
            remove(temp_file.name)

        if is_json or is_csv:
            # for parquet file the time filtering is executed in `reader`
            df = filter_df_start_end_time(
                df,
                time_column=time_column,
                start_time=start_time,
                end_time=end_time,
            )
            if drop_time_column:
                df.drop(columns=[time_column], inplace=True)
        if is_json:
            # for csv and parquet files the columns select is executed in `reader`.
            df = select_columns_from_df(df, columns=columns)
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

    def delete(self):
        """delete the object from the datastore"""
        self._store.rm(self._path)

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
        """get the local path of the file, download to tmp first if it's a remote object"""
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

    def remove_local(self):
        """remove the local file if it exists and was downloaded from a remote object"""
        if self.kind == "file":
            return

        if self._local_path:
            remove(self._local_path)
            self._local_path = ""

    def as_df(
        self,
        columns=None,
        df_module=None,
        format="",
        time_column=None,
        start_time=None,
        end_time=None,
        **kwargs,
    ):
        """return a dataframe object (generated from the dataitem).

        :param columns:     optional, list of columns to select
        :param df_module:   optional, py module used to create the DataFrame (e.g. pd, dd, cudf, ..)
        :param format:      file format, if not specified it will be deducted from the suffix
        :param start_time:  filters out data before this time
        :param end_time:    filters out data after this time
        :param time_column: Store timestamp_key will be used if None.
                            The results will be filtered by this column and start_time & end_time.
        """
        df = self._store.as_df(
            self._url,
            self._path,
            columns=columns,
            df_module=df_module,
            format=format,
            time_column=time_column,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )
        return df

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

    def get_artifact_type(self) -> Union[str, None]:
        """
        Check if the data item represents an Artifact (one of Artifact, DatasetArtifact and ModelArtifact). If it does
        it return the store uri prefix (artifacts, datasets or models), otherwise None.

        :return: The store prefix of the artifact if it is an artifact data item and None if not.
        """
        if self.artifact_url and is_store_uri(url=self.artifact_url):
            store_uri_prefix = parse_store_uri(self.artifact_url)[0]
            if StorePrefix.is_artifact(prefix=store_uri_prefix):
                return store_uri_prefix
        return None

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
        raise OSError(f"error: cannot connect to {url}: {err_to_str(exc)}")

    mlrun.errors.raise_for_status(response)

    return response.content


def http_head(url, headers=None, auth=None):
    try:
        response = requests.head(url, headers=headers, auth=auth, verify=verify_ssl)
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {err_to_str(exc)}")

    mlrun.errors.raise_for_status(response)

    return response.headers


def http_put(url, data, headers=None, auth=None):
    try:
        response = requests.put(
            url, data=data, headers=headers, auth=auth, verify=verify_ssl
        )
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {err_to_str(exc)}")

    mlrun.errors.raise_for_status(response)


def http_upload(url, file_path, headers=None, auth=None):
    with open(file_path, "rb") as data:
        http_put(url, data, headers, auth)


class HttpStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets)
        self._https_auth_token = None
        self._schema = schema
        self.auth = None
        self._headers = {}
        self._enrich_https_token()
        self._validate_https_token()

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
        data = http_get(self.url + self._join(key), self._headers, self.auth)
        if offset:
            data = data[offset:]
        if size:
            data = data[:size]
        return data

    def _enrich_https_token(self):
        token = self._get_secret_or_env("HTTPS_AUTH_TOKEN")
        if token:
            self._https_auth_token = token
            self._headers.setdefault("Authorization", f"token {token}")

    def _validate_https_token(self):
        if self._https_auth_token and self._schema in ["http"]:
            logger.warn(
                f"A AUTH TOKEN should not be provided while using {self._schema} "
                f"schema as it is not secure and is not recommended."
            )
