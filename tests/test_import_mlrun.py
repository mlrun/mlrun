# Copyright 2025 Iguazio
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

import subprocess

acceptable_stderr_errors = [
    "Kubeflow Pipelines (KFP) is not installed. Using noop implementations."
]


def test_import_mlrun():
    out = subprocess.run(["python", "-c", "import mlrun"], capture_output=True)
    stdout_lines = out.stdout.decode().strip().split("\n")
    stderr_lines = out.stderr.decode().strip().split("\n")
    unexpected_stdout_errors = [line for line in stdout_lines if "[error]" in line]
    unexpected_stderr_errors = [
        line for line in stderr_lines if line and line not in acceptable_stderr_errors
    ]
    assert unexpected_stdout_errors == [], "`import mlrun` wrote unexpected error logs"
    assert (
        unexpected_stderr_errors == []
    ), "`import mlrun` wrote unexpected errors to stderr"
