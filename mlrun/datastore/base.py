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
from urllib.parse import urlparse

import fsspec
import orjson
import pandas as pd
import pyarrow
import pytz
import requests
from deprecated import deprecated

import mlrun.config
import mlrun.errors
from mlrun.errors import err_to_str
from mlrun.utils import StorePrefix, is_jupyter, logger

from .store_resources import is_store_uri, parse_store_uri
from .utils import filter_df_start_end_time, select_columns_from_df


class FileStats:
    def __init__(self, size, modified, content_type=None):
        self.size = size
        self.modified = modified
        self.content_type = content_type

    def __repr__(self):
        return f"FileStats(size={self.size}, modified={self.modified}, type={self.content_type})"


class DataStore:
    using_bucket = False

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
    def _sanitize_storage_options(options):
        if not options:
            return {}
        options = {k: v for k, v in options.items() if v is not None and v != ""}
        return options

    @staticmethod
    def _sanitize_url(url):
        """
        Extract only the schema, netloc, and path from an input URL if they exist,
        excluding parameters, query, or fragments.
        """
        if not url:
            raise mlrun.errors.MLRunInvalidArgumentError("Cannot parse an empty URL")
        parsed_url = urllib.parse.urlparse(url)
        netloc = f"//{parsed_url.netloc}" if parsed_url.netloc else "//"
        return f"{parsed_url.scheme}:{netloc}{parsed_url.path}"

    @staticmethod
    def uri_to_kfp(endpoint, subpath):
        raise ValueError("data store doesnt support KFP URLs")

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return ""

    # TODO: remove in 1.8.0
    @deprecated(
        version="1.8.0",
        reason="'get_filesystem()' will be removed in 1.8.0, use "
        "'filesystem' property instead",
        category=FutureWarning,
    )
    def get_filesystem(self):
        return self.filesystem

    @property
    def filesystem(self) -> Optional[fsspec.AbstractFileSystem]:
        """return fsspec file system object, if supported"""
        return None

    def supports_isdir(self):
        """Whether the data store supports isdir"""
        return True

    def _get_secret_or_env(self, key, default=None, prefix=None):
        # Project-secrets are mounted as env variables whose name can be retrieved from SecretsStore
        return mlrun.get_secret_or_env(
            key, secret_provider=self._get_secret, default=default, prefix=prefix
        )

    def get_storage_options(self):
        """get fsspec storage options"""
        return self._sanitize_storage_options(None)

    def open(self, filepath, mode):
        file_system = self.filesystem
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

    @property
    def spark_url(self):
        return self.url

    def get(self, key, size=None, offset=0):
        pass

    def query(self, key, query="", **kwargs):
        raise ValueError("data store doesnt support structured queries")

    def put(self, key, data, append=False):
        pass

    def _prepare_put_data(self, data, append=False):
        mode = "a" if append else "w"
        if isinstance(data, bytearray):
            data = bytes(data)

        if isinstance(data, bytes):
            return data, f"{mode}b"
        elif isinstance(data, str):
            return data, mode
        else:
            raise TypeError(f"Unable to put a value of type {type(self).__name__}")

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

    def get_spark_options(self):
        return {}

    @staticmethod
    def _parquet_reader(
        df_module,
        url,
        file_system,
        time_column,
        start_time,
        end_time,
        additional_filters,
    ):
        from storey.utils import find_filters, find_partitions

        def set_filters(
            partitions_time_attributes,
            start_time_inner,
            end_time_inner,
            filters_inner,
            kwargs,
        ):
            filters = []
            find_filters(
                partitions_time_attributes,
                start_time_inner,
                end_time_inner,
                filters,
                time_column,
            )
            if filters and filters_inner:
                filters[0] += filters_inner

            kwargs["filters"] = filters

        def reader(*args, **kwargs):
            if time_column is None and (start_time or end_time):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "When providing start_time or end_time, must provide time_column"
                )
            if (
                start_time
                and end_time
                and start_time.utcoffset() != end_time.utcoffset()
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "start_time and end_time must have the same time zone"
                )

            if start_time or end_time or additional_filters:
                partitions_time_attributes = find_partitions(url, file_system)
                set_filters(
                    partitions_time_attributes,
                    start_time,
                    end_time,
                    additional_filters,
                    kwargs,
                )
                try:
                    return df_module.read_parquet(*args, **kwargs)
                except pyarrow.lib.ArrowInvalid as ex:
                    if not str(ex).startswith(
                        "Cannot compare timestamp with timezone to timestamp without timezone"
                    ):
                        raise ex

                    start_time_inner = None
                    if start_time:
                        start_time_inner = start_time.replace(
                            tzinfo=None if start_time.tzinfo else pytz.utc
                        )

                    end_time_inner = None
                    if end_time:
                        end_time_inner = end_time.replace(
                            tzinfo=None if end_time.tzinfo else pytz.utc
                        )

                    set_filters(
                        partitions_time_attributes,
                        start_time_inner,
                        end_time_inner,
                        additional_filters,
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
        additional_filters=None,
        **kwargs,
    ):
        df_module = df_module or pd
        file_url = self._sanitize_url(url)
        is_csv, is_json, drop_time_column = False, False, False
        file_system = self.filesystem
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
                        if df_module is pd:
                            kwargs.pop("filesystem", None)
                            kwargs.pop("storage_options", None)
                            for filename in filenames:
                                fullpath = f"{base_path}/{filename}"
                                with file_system.open(fullpath) as fhandle:
                                    updated_args = [fhandle]
                                    updated_args.extend(args[1:])
                                    dfs.append(
                                        df_module.read_csv(*updated_args, **kwargs)
                                    )
                        else:
                            for filename in filenames:
                                updated_args = [f"{base_path}/{filename}"]
                                updated_args.extend(args[1:])
                                dfs.append(df_module.read_csv(*updated_args, **kwargs))
                        return df_module.concat(dfs)

        elif mlrun.utils.helpers.is_parquet_file(file_url, format):
            if columns:
                kwargs["columns"] = columns

            reader = self._parquet_reader(
                df_module,
                url,
                file_system,
                time_column,
                start_time,
                end_time,
                additional_filters,
            )

        elif file_url.endswith(".json") or format == "json":
            is_json = True
            reader = df_module.read_json

        else:
            raise Exception(f"File type unhandled {url}")

        if file_system:
            storage_options = self.get_storage_options()
            if url.startswith("ds://"):
                parsed_url = urllib.parse.urlparse(url)
                url = parsed_url.path
                if self.using_bucket:
                    url = url[1:]
                # Pass the underlying file system
                kwargs["filesystem"] = file_system
            elif storage_options:
                kwargs["storage_options"] = storage_options
            df = reader(url, **kwargs)
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
        try:
            self.filesystem.rm(path=path, recursive=recursive, maxdepth=maxdepth)
        except FileNotFoundError:
            pass

    @staticmethod
    def _is_dd(df_module):
        try:
            import dask.dataframe as dd

            return df_module == dd
        except ImportError:
            return False


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
        train_run = train_iris_func.run(
            inputs={"dataset": dataset}, params={"label_column": "label"}
        )

        train_run.artifact("confusion-matrix").show()
        test_set = train_run.artifact("test_set").as_df()

        # create and use DataItem from uri
        data = mlrun.get_dataitem("http://xyz/data.json").get()
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
        additional_filters=None,
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
        :param additional_filters: List of additional_filter conditions as tuples.
                                    Each tuple should be in the format (column_name, operator, value).
                                    Supported operators: "=", ">=", "<=", ">", "<".
                                    Example: [("Product", "=", "Computer")]
                                    For all supported filters, please see:
                                    https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
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
            additional_filters=additional_filters,
            **kwargs,
        )
        return df

    def show(self, format: Optional[str] = None) -> None:
        """show the data object content in Jupyter

        :param format: format to use (when there is no/wrong suffix), e.g. 'png'
        """
        if not is_jupyter:
            logger.warning(
                "Jupyter was not detected. `.show()` displays only inside Jupyter."
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


class HttpStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets)
        self._https_auth_token = None
        self._schema = schema
        self.auth = None
        self._headers = {}
        self._enrich_https_token()
        self._validate_https_token()

    @property
    def filesystem(self):
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
        data = self._http_get(self.url + self._join(key), self._headers, self.auth)
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

    def _http_get(
        self,
        url,
        headers=None,
        auth=None,
    ):
        # import here to prevent import cycle
        from mlrun.config import config as mlconf

        verify_ssl = mlconf.httpdb.http.verify
        try:
            response = requests.get(url, headers=headers, auth=auth, verify=verify_ssl)
        except OSError as exc:
            raise OSError(f"error: cannot connect to {url}: {err_to_str(exc)}")

        mlrun.errors.raise_for_status(response)
        return response.content


# This wrapper class is designed to extract the 'ds' schema and profile name from URL-formatted paths.
# Within fsspec, the AbstractFileSystem::_strip_protocol() internal method is used to handle complete URL paths.
# As an example, it converts an S3 URL 's3://s3bucket/path' to just 's3bucket/path'.
# Since 'ds' schemas are not inherently processed by fsspec, we have adapted the _strip_protocol()
# method specifically to strip away the 'ds' schema as required.
def make_datastore_schema_sanitizer(cls, using_bucket=False, *args, **kwargs):
    if not issubclass(cls, fsspec.AbstractFileSystem):
        raise ValueError("Class must be a subclass of fsspec.AbstractFileSystem")

    class DatastoreSchemaSanitizer(cls):
        @classmethod
        def _strip_protocol(cls, url):
            if url.startswith("ds://"):
                parsed_url = urlparse(url)
                url = parsed_url.path
                if using_bucket:
                    url = url[1:]
            return super()._strip_protocol(url)

    return DatastoreSchemaSanitizer(*args, **kwargs)
