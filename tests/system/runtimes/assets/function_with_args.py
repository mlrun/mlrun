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

import argparse
import sys

import mlrun.artifacts
from mlrun import get_or_create_ctx

parser = argparse.ArgumentParser()
parser.add_argument("--some-arg")
parser.add_argument("--another-one", required=False)


def handler(context):
    # need to parse with unknown args since handlers are called from within mlrun code and not from the command line
    flags, unknown = parser.parse_known_args()
    some_arg = flags.some_arg
    context.log_result("some-arg-by-handler", some_arg)
    context.log_result("my-args", sys.argv)


def handler_with_future_links(
    context, p1: int
) -> tuple[mlrun.artifacts.ModelArtifact, int]:
    context.log_artifact("some_file", body=b"abc is 123", local_path="my_file.txt")
    my_model = context.log_model(
        "my_model",
        body=b"abc is 123",
        model_file="model.txt",
        metrics={"accuracy": 0.85},
        parameters={"xx": "abc"},
        labels={"framework": "xgboost"},
        artifact_path=context.artifact_subpath("models"),
        extra_data={"some_file": ..., "px": ...},
    )

    return my_model, p1


if __name__ == "__main__":
    flags = parser.parse_args()
    some_arg = flags.some_arg
    another_one = flags.another_one

    job_name = "function-with-args"
    context = get_or_create_ctx(job_name)
    context.log_result("some-arg-by-main", some_arg)
    if another_one:
        context.log_result("another-one", another_one)

    context.log_result("my-args", sys.argv)
