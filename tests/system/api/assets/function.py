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
#
import os


def secret_test_function(context, secrets: list = None):
    """Validate that given secrets exists

    :param context: the MLRun context
    :param secrets: name of the secrets that we want to look at
    """
    context.logger.info("running function")
    secrets = secrets or []
    for sec_name in secrets:
        sec_value = context.get_secret(sec_name)
        context.logger.info(f"Secret: {sec_name} ==> {sec_value}")
        context.log_result(sec_name, sec_value)
    return True


def log_artifact_test_function(context, body_size: int = 1000, inline: bool = True):
    """Logs artifact given its event body
    :param context: the MLRun context
    :param body_size: size of the artifact body
    :param inline: whether to log the artifact body inline or not
    """
    context.logger.info("running function")
    body = b"a" * body_size
    context.log_artifact("test", body=body, is_inline=inline)
    context.logger.info("run complete!", body_len=len(body))
    return True


def log_artifact_many_tags(context):
    body = b"abc123"
    context.log_artifact("file_result", body=body, tag="v1")
    context.log_artifact("file_result", body=body, tag="v2")
    context.log_artifact("file_result", body=body, tag="v3")


def log_artifact_with_tag(context, tag):
    context.log_artifact("file_result", body=b"abc123", tag=tag)


def access_key_verifier(context, v3io_access_key: str):
    assert os.environ.get("V3IO_ACCESS_KEY") == v3io_access_key
