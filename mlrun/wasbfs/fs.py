import re

from adlfs import AzureBlobFileSystem


class WasbFS(AzureBlobFileSystem):

    protocol = "wasb"

    def _convert_wasb_schema_to_az(self, url):
        # wasbs pattern: wasbs://<CONTAINER>@<ACCOUNT_NAME>.blob.core.windows.net/<PATH_OBJ_IN_CONTAINER>
        if url.startswith("wasb://") or url.startswith("wasbs://"):
            m = re.match(
                r"^(?P<schema>.*)://(?P<cont>.*)@(?P<account>.*?)\..*?/(?P<obj_path>.*?)$",
                url,
            )
            retval = "az://" + m.groupdict()["cont"] + "/" + m.groupdict()["obj_path"]
        else:
            m = re.match(
                r"^(?P<cont>.*)@(?P<account>.*?)\..*?/(?P<obj_path>.*?)$",
                url,
            )
            retval = m.groupdict()["cont"] + "/" + m.groupdict()["obj_path"]

        return retval

    def info(self, path, refresh=False, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().info(conv_path, refresh, **kwargs)

    def glob(self, path, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().glob(conv_path, **kwargs)

    def ls(
        self,
        path: str,
        detail: bool = False,
        invalidate_cache: bool = False,
        delimiter: str = "/",
        return_glob: bool = False,
        **kwargs,
    ):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().ls(
            conv_path,
            detail,
            invalidate_cache,
            delimiter,
            return_glob,
            **kwargs,
        )

    def find(self, path, withdirs=False, prefix="", **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().find(conv_path, withdirs, prefix, **kwargs)

    def mkdir(self, path, exist_ok=False):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().mkdir(conv_path, exist_ok)

    def rmdir(self, path: str, delimiter="/", **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().rmdir(conv_path, delimiter, **kwargs)

    def size(self, path):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().size(conv_path)

    def isfile(self, path):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().isfile(conv_path)

    def isdir(self, path):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().isdir(conv_path)

    def exists(self, path):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().exists(conv_path)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().cat(conv_path, recursive, on_error, **kwargs)

    def url(self, path, expires=3600, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().url(conv_path, expires, **kwargs)

    def expand_path(self, path, recursive=False, maxdepth=None):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().expand_path(conv_path, recursive, maxdepth)

    def rm(self, path, recursive=False, maxdepth=None, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().rm(conv_path, recursive, maxdepth, **kwargs)

    def open(self, path, mode="rb", block_size=None, cache_options=None, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().open(conv_path, mode, block_size, cache_options, **kwargs)

    def touch(self, path, truncate=True, **kwargs):
        conv_path = self._convert_wasb_schema_to_az(path)
        return super().touch(conv_path, truncate, **kwargs)
