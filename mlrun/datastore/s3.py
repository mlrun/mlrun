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

import time

import boto3

import mlrun.errors

from .base import DataStore, FileStats, get_range
from .datastore_profile import datastore_profile_read


class S3Store(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets)
        # will be used in case user asks to assume a role and work through fsspec
        self._temp_credentials = None
        region = None

        self.headers = None

        if schema == "ds":
            datastore_profile = datastore_profile_read(name)
            access_key = datastore_profile.access_key
            secret_key = datastore_profile.secret_key
            endpoint_url = datastore_profile.endpoint_url
            force_non_anonymous = datastore_profile.force_non_anonymous
            profile_name = datastore_profile.profile_name
            assume_role_arn = datastore_profile.assume_role_arn
            self.endpoint = ""
        else:
            access_key = self._get_secret_or_env("AWS_ACCESS_KEY_ID")
            secret_key = self._get_secret_or_env("AWS_SECRET_ACCESS_KEY")
            endpoint_url = self._get_secret_or_env("S3_ENDPOINT_URL")
            force_non_anonymous = self._get_secret_or_env("S3_NON_ANONYMOUS")
            profile_name = self._get_secret_or_env("AWS_PROFILE")
            assume_role_arn = self._get_secret_or_env("MLRUN_AWS_ROLE_ARN")

        # If user asks to assume a role, this needs to go through the STS client and retrieve temporary creds
        if assume_role_arn:
            client = boto3.client(
                "sts", aws_access_key_id=access_key, aws_secret_access_key=secret_key
            )
            self._temp_credentials = client.assume_role(
                RoleArn=assume_role_arn, RoleSessionName="assumeRoleSession"
            ).get("Credentials")
            if not self._temp_credentials:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"cannot assume role {assume_role_arn}"
                )

            self.s3 = boto3.resource(
                "s3",
                region_name=region,
                aws_access_key_id=self._temp_credentials["AccessKeyId"],
                aws_secret_access_key=self._temp_credentials["SecretAccessKey"],
                aws_session_token=self._temp_credentials["SessionToken"],
                endpoint_url=endpoint_url,
            )
            return

        # User asked for a profile to be used. We don't use access-key or secret-key for this, since the
        # parameters should be in the ~/.aws/credentials file for this to work
        if profile_name:
            session = boto3.session.Session(profile_name=profile_name)
            self.s3 = session.resource(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url,
            )
            return

        if access_key or secret_key or force_non_anonymous:
            self.s3 = boto3.resource(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint_url,
            )
        else:
            # from env variables
            self.s3 = boto3.resource(
                "s3", region_name=region, endpoint_url=endpoint_url
            )
            # If not using credentials, boto will still attempt to sign the requests, and will fail any operations
            # due to no credentials found. These commands disable signing and allow anonymous mode (same as
            # anon in the storage_options when working with fsspec).
            from botocore.handlers import disable_signing

            self.s3.meta.client.meta.events.register(
                "choose-signer.s3.*", disable_signing
            )

    def get_filesystem(self, silent=False):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            # noqa
            from mlrun.datastore.s3fs_store import S3FileSystemWithDS
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    "AWS s3fs not installed, run pip install s3fs"
                ) from exc
            return None

        self._filesystem = S3FileSystemWithDS(**self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        if self.kind == "ds":
            # If it's a datastore profile, 'self.name' holds the URL path to the item, e.g.,
            # 'ds://some_profile/s3bucket/path/to/object'
            # The function 'datastore_profile_read()' derives the profile name from this URL,
            # reads the profile, and fetches the credentials.
            datastore_profile = datastore_profile_read(self.name)
            endpoint_url = datastore_profile.endpoint_url
            force_non_anonymous = datastore_profile.force_non_anonymous
            profile = datastore_profile.profile_name
            key = datastore_profile.access_key
            secret = datastore_profile.secret_key
        else:
            force_non_anonymous = self._get_secret_or_env("S3_NON_ANONYMOUS")
            profile = self._get_secret_or_env("AWS_PROFILE")
            endpoint_url = self._get_secret_or_env("S3_ENDPOINT_URL")
            key = self._get_secret_or_env("AWS_ACCESS_KEY_ID")
            secret = self._get_secret_or_env("AWS_SECRET_ACCESS_KEY")

        if self._temp_credentials:
            key = self._temp_credentials["AccessKeyId"]
            secret = self._temp_credentials["SecretAccessKey"]
            token = self._temp_credentials["SessionToken"]
        else:
            token = None

        storage_options = dict(
            anon=not (force_non_anonymous or (key and secret)),
            key=key,
            secret=secret,
            token=token,
        )

        if endpoint_url:
            client_kwargs = {"endpoint_url": endpoint_url}
            storage_options["client_kwargs"] = client_kwargs

        if profile:
            storage_options["profile"] = profile

        return storage_options

    def get_bucket_and_key(self, key):
        path = self._join(key)[1:]
        if self.endpoint:
            return self.endpoint, path
        directories = path.split("/")
        bucket = directories[0]
        return bucket, path[len(bucket) + 1 :]

    def upload(self, key, src_path):
        bucket, key = self.get_bucket_and_key(key)
        self.s3.Object(bucket, key).put(Body=open(src_path, "rb"))

    def get(self, key, size=None, offset=0):
        bucket, key = self.get_bucket_and_key(key)
        obj = self.s3.Object(bucket, key)
        if size or offset:
            return obj.get(Range=get_range(size, offset))["Body"].read()
        return obj.get()["Body"].read()

    def put(self, key, data, append=False):
        bucket, key = self.get_bucket_and_key(key)
        self.s3.Object(bucket, key).put(Body=data)

    def stat(self, key):
        bucket, key = self.get_bucket_and_key(key)
        obj = self.s3.Object(bucket, key)
        size = obj.content_length
        modified = obj.last_modified
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        bucket, key = self.get_bucket_and_key(key)
        if not key.endswith("/"):
            key += "/"
        # Object names is S3 are not fully following filesystem semantics - they do not start with /, even for
        # "absolute paths". Therefore, we are are removing leading / from path filter.
        if key.startswith("/"):
            key = key[1:]
        key_length = len(key)
        bucket = self.s3.Bucket(bucket)
        return [obj.key[key_length:] for obj in bucket.objects.filter(Prefix=key)]


def parse_s3_bucket_and_key(s3_path):
    try:
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
    except Exception as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "failed to parse s3 bucket and key"
        ) from exc

    return bucket, key
