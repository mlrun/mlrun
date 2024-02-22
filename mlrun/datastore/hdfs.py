import os

import fsspec

from mlrun.datastore.base import DataStore


class HdfsStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets)

        self.host = self._get_secret_or_env("HDFS_HOST")
        self.port = self._get_secret_or_env("HDFS_PORT")
        self.http_port = self._get_secret_or_env("HDFS_HTTP_PORT")
        self.user = self._get_secret_or_env("HDFS_USER")
        if not self.user:
            self.user = os.environ.get("HADOOP_USER_NAME", os.environ.get("USER"))

        self._filesystem = None

    @property
    def filesystem(self):
        if not self._filesystem:
            self._filesystem = fsspec.filesystem(
                "webhdfs",
                host=self.host,
                port=self.http_port,
                user=self.user,
            )
        return self._filesystem

    @property
    def url(self):
        return f"webhdfs://{self.host}:{self.http_port}"

    @property
    def spark_url(self):
        return f"hdfs://{self.host}:{self.port}"
