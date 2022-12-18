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
#

import argparse
import sys

from mlrun import get_or_create_ctx

parser = argparse.ArgumentParser()
parser.add_argument("--some-arg")


def handler(context):
    # need to parse with unknown args since handlers are called from within mlrun code and not from the command line
    flags, unknown = parser.parse_known_args()
    some_arg = flags.some_arg
    context.log_result("some-arg-by-handler", some_arg)
    context.log_result("my-args", sys.argv)


if __name__ == "__main__":
    flags = parser.parse_args()
    some_arg = flags.some_arg

    job_name = "function-with-args"
    context = get_or_create_ctx(job_name)
    context.log_result("some-arg-by-main", some_arg)
    context.log_result("my-args", sys.argv)
