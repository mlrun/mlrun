#!/usr/bin/env python

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
import pathlib
from os import environ, path
from subprocess import Popen
from sys import executable
from urllib.parse import urlparse

import click
import dotenv

import mlrun
from mlrun.config import config as mlconf


@click.group()
def main():
    pass


@main.command()
@click.option("--port", "-p", help="port to listen on", type=int)
@click.option("--dirpath", "-d", help="database directory (dirpath)")
@click.option("--dsn", "-s", help="database dsn, e.g. sqlite:///db/mlrun.db")
@click.option("--logs-path", "-l", help="logs directory path")
@click.option("--data-volume", "-v", help="path prefix to the location of artifacts")
@click.option("--verbose", is_flag=True, help="verbose log")
@click.option("--background", "-b", is_flag=True, help="run in background process")
@click.option("--artifact-path", "-a", help="default artifact path")
@click.option(
    "--update-env",
    default="",
    is_flag=False,
    flag_value=mlrun.config.default_env_file,
    help=f"update the specified mlrun .env file (if TEXT not provided defaults to {mlrun.config.default_env_file})",
)
def db(
    port,
    dirpath,
    dsn,
    logs_path,
    data_volume,
    verbose,
    background,
    artifact_path,
    update_env,
):
    """Run HTTP api/database server"""
    env = environ.copy()
    # ignore client side .env file (so import mlrun in server will not try to connect to local/remote DB)
    env["MLRUN_IGNORE_ENV_FILE"] = "true"
    env["MLRUN_DBPATH"] = ""

    if port is not None:
        env["MLRUN_httpdb__port"] = str(port)
    if dirpath is not None:
        env["MLRUN_httpdb__dirpath"] = dirpath
    if dsn is not None:
        if dsn.startswith("sqlite://") and "check_same_thread=" not in dsn:
            dsn += "?check_same_thread=false"
        env["MLRUN_HTTPDB__DSN"] = dsn
    if logs_path is not None:
        env["MLRUN_HTTPDB__LOGS_PATH"] = logs_path
    if data_volume is not None:
        env["MLRUN_HTTPDB__DATA_VOLUME"] = data_volume
    if verbose:
        env["MLRUN_LOG_LEVEL"] = "DEBUG"
    if artifact_path or "MLRUN_ARTIFACT_PATH" not in env:
        if not artifact_path:
            artifact_path = (
                env.get("MLRUN_HTTPDB__DATA_VOLUME", "./artifacts").rstrip("/")
                + "/{{project}}"
            )
        env["MLRUN_ARTIFACT_PATH"] = path.realpath(path.expanduser(artifact_path))

    env["MLRUN_IS_API_SERVER"] = "true"

    # create the DB dir if needed
    dsn = dsn or mlconf.httpdb.dsn
    if dsn and dsn.startswith("sqlite:///"):
        parsed = urlparse(dsn)
        p = pathlib.Path(parsed.path[1:]).parent
        p.mkdir(parents=True, exist_ok=True)

    cmd = [executable, "-m", "server.api.main"]
    if env.get("MLRUN_MEMRAY") != "0":
        cmd = [executable, "-m", "memray", "run"]
        output_file = env.get("MLRUN_MEMRAY_OUTPUT_FILE", None)
        if output_file:
            cmd += ["--output", output_file, "--force"]

        cmd += ["-m", "server.api.main"]

    pid = None
    if background:
        print("Starting MLRun API service in the background...")
        child = Popen(
            cmd,
            env=env,
            stdout=open("mlrun-stdout.log", "w"),
            stderr=open("mlrun-stderr.log", "w"),
            start_new_session=True,
        )
        pid = child.pid
        print(
            f"background pid: {pid}, logs written to mlrun-stdout.log and mlrun-stderr.log, use:\n"
            f"`kill {pid}` (linux/mac) or `taskkill /pid {pid} /t /f` (windows), to kill the mlrun service process"
        )
    else:
        child = Popen(cmd, env=env)
        returncode = child.wait()
        if returncode != 0:
            raise SystemExit(returncode)
    if update_env:
        # update mlrun client env file with the API path, so client will use the new DB
        # update and PID, allow killing the correct process in a config script
        filename = path.expanduser(update_env)
        dotenv.set_key(
            filename, "MLRUN_DBPATH", f"http://localhost:{port or 8080}", quote_mode=""
        )
        dotenv.set_key(filename, "MLRUN_MOCK_NUCLIO_DEPLOYMENT", "auto", quote_mode="")
        if pid:
            dotenv.set_key(filename, "MLRUN_SERVICE_PID", str(pid), quote_mode="")
        print(f"updated configuration in {update_env} .env file")


if __name__ == "__main__":
    main()
