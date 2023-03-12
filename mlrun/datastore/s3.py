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

import time

import boto3
import fsspec

import mlrun.errors

from .base import DataStore, FileStats, get_range


class S3Store(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets)
        # will be used in case user asks to assume a role and work through fsspec
        self._temp_credentials = None
        region = None

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
            import s3fs  # noqa
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    "AWS s3fs not installed, run pip install s3fs"
                ) from exc
            return None
        self._filesystem = fsspec.filesystem("s3", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        if self._temp_credentials:
            key = self._temp_credentials["AccessKeyId"]
            secret = self._temp_credentials["SecretAccessKey"]
            token = self._temp_credentials["SessionToken"]
        else:
            key = self._get_secret_or_env("AWS_ACCESS_KEY_ID")
            secret = self._get_secret_or_env("AWS_SECRET_ACCESS_KEY")
            token = None

        force_non_anonymous = self._get_secret_or_env("S3_NON_ANONYMOUS")
        profile = self._get_secret_or_env("AWS_PROFILE")

        storage_options = dict(
            anon=not (force_non_anonymous or (key and secret)),
            key=key,
            secret=secret,
            token=token,
        )

        endpoint_url = self._get_secret_or_env("S3_ENDPOINT_URL")
        if endpoint_url:
            client_kwargs = {"endpoint_url": endpoint_url}
            storage_options["client_kwargs"] = client_kwargs

        if profile:
            storage_options["profile"] = profile

        return storage_options

    def upload(self, key, src_path):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(
            Body=open(src_path, "rb")
        )

    def get(self, key, size=None, offset=0):
        obj = self.s3.Object(self.endpoint, self._join(key)[1:])
        if size or offset:
            return obj.get(Range=get_range(size, offset))["Body"].read()
        return obj.get()["Body"].read()

    def put(self, key, data, append=False):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(Body=data)

    def stat(self, key):
        obj = self.s3.Object(self.endpoint, self._join(key)[1:])
        size = obj.content_length
        modified = obj.last_modified
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        if not key.endswith("/"):
            key += "/"
        # Object names is S3 are not fully following filesystem semantics - they do not start with /, even for
        # "absolute paths". Therefore, we are are removing leading / from path filter.
        if key.startswith("/"):
            key = key[1:]
        key_length = len(key)
        bucket = self.s3.Bucket(self.endpoint)
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
