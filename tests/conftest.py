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

import traceback
import typing
from datetime import datetime
from http import HTTPStatus
from os import environ
from pathlib import Path
from subprocess import PIPE, run
from sys import executable, platform, stderr
from time import monotonic, sleep
from urllib.request import URLError, urlopen

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

tests_root_directory = Path(__file__).absolute().parent
results = tests_root_directory / "test_results"
is_ci = "CI" in environ

environ["KFPMETA_OUT_DIR"] = f"{results}/kfp/"
environ["KFP_ARTIFACTS_DIR"] = f"{results}/kfp/"

rundb_path = f"{results}/rundb"
logs_path = f"{results}/logs"
out_path = f"{results}/out"
root_path = str(Path(tests_root_directory).parent)
examples_path = Path(tests_root_directory).parent.joinpath("examples")
pytest_plugins = ["tests.common_fixtures"]

# import package stuff after setting env vars so it will take effect
from mlrun.api.db.sqldb.db import run_time_fmt  # noqa: E402
from mlrun.api.db.sqldb.models import Base  # noqa: E402


def check_docker():
    if not platform.startswith("linux"):
        return False

    with open("/proc/1/cgroup") as fp:
        for line in fp:
            if "/docker/" in line:
                return True
    return False


in_docker = check_docker()

# This must be *after* environment changes above
from mlrun import RunObject, RunTemplate  # noqa


def tag_test(spec: RunTemplate, name) -> RunTemplate:
    spec = spec.copy()
    spec.metadata.name = name
    spec.metadata.labels["test"] = name
    return spec


def has_secrets():
    return Path("secrets.txt").is_file()


def verify_state(result: RunObject):
    state = result.status.state
    assert state == "completed", f"wrong state ({state}) {result.status.error}"


def wait_for_server(url, timeout_sec):
    start = monotonic()
    while monotonic() - start <= timeout_sec:
        try:
            with urlopen(url) as resp:
                if resp.status == HTTPStatus.OK.value:
                    return True
        except (URLError, ConnectionError):
            pass
        sleep(0.1)
    return False


def run_now():
    return datetime.now().strftime(run_time_fmt)


def new_run(state, labels, uid=None, **kw):
    obj = {
        "metadata": {"name": "run-name", "labels": labels},
        "status": {"state": state, "start_time": run_now()},
    }
    if uid:
        obj["metadata"]["uid"] = uid
    obj.update(kw)
    return obj


def freeze(f, **kwargs):
    """
    Enables to override an attribute passed to a sub-function without the need to access the function directly
    :param f: the function we want to pass the attribute to
    :param kwargs: dictionary containing name(key) and value of the attributes to override.
    :return: wrapped function with overridden attributes
    """
    frozen = kwargs

    def wrapper(*args, **kwargs):
        kwargs.update(frozen)
        return f(*args, **kwargs)

    return wrapper


def init_sqldb(dsn):
    engine = create_engine(dsn)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def exec_mlrun(args, cwd=None, op="run"):
    cmd = [executable, "-m", "mlrun", op] + args
    out = run(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=stderr)
        print(out.stdout.decode("utf-8"), file=stderr)
        print(traceback.format_exc())
        raise Exception(out.stderr.decode("utf-8"))
    return out.stdout.decode("utf-8")


class MockSpecificCalls:
    def __init__(
        self,
        original_function: typing.Callable,
        call_indexes_to_mock: typing.List[int],
        return_value: typing.Any,
    ):
        self.original_function = original_function
        self.call_indexes_to_mock = call_indexes_to_mock
        self.return_value = return_value

    calls = 0

    def mock_function(self, *args, **kwargs):
        self.calls += 1
        if self.calls not in self.call_indexes_to_mock:
            return self.original_function(*args, **kwargs)
        else:
            return self.return_value
