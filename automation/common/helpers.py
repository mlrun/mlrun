# Copyright 2023 MLRun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import typing


def run_command(
    command: str,
    args: list = None,
    workdir: str = None,
    stdin: str = None,
    live: bool = True,
    log_file_handler: typing.IO[str] = None,
) -> (str, str, int):
    if workdir:
        command = f"cd {workdir}; " + command
    if args:
        command += " " + " ".join(args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        shell=True,
    )

    if stdin:
        process.stdin.write(bytes(stdin, "ascii"))
        process.stdin.close()

    stdout = _handle_command_stdout(process.stdout, log_file_handler, live)
    stderr = process.stderr.read()
    exit_status = process.wait()

    return stdout, stderr, exit_status


def _handle_command_stdout(
    stdout_stream: typing.IO[bytes],
    log_file_handler: typing.IO[str] = None,
    live: bool = True,
) -> str:
    def _write_to_log_file(text: bytes):
        if log_file_handler:
            log_file_handler.write(text.decode(sys.stdout.encoding))

    stdout = ""
    if live:
        for line in iter(stdout_stream.readline, b""):
            stdout += str(line)
            sys.stdout.write(line.decode(sys.stdout.encoding))
            _write_to_log_file(line)
    else:
        stdout = stdout_stream.read()
        _write_to_log_file(stdout)

    return stdout
