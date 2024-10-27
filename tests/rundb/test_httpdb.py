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

import codecs
import datetime
import sys
import time
from collections import namedtuple
from os import environ
from pathlib import Path
from shutil import rmtree
from socket import socket
from subprocess import DEVNULL, PIPE, Popen, run
from sys import executable
from tempfile import mkdtemp
from uuid import uuid4

import deepdiff
import pytest
import requests_mock as requests_mock_package

import mlrun.alerts
import mlrun.artifacts.base
import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.errors
import mlrun.projects.project
from mlrun import RunObject
from mlrun.db.auth_utils import StaticTokenProvider
from mlrun.db.httpdb import HTTPRunDB
from tests.conftest import tests_root_directory, wait_for_server

project_dir_path = Path(__file__).absolute().parent.parent.parent
Server = namedtuple("Server", "url conn workdir")

docker_tag = "mlrun/test-api"


def free_port():
    with socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def check_server_up(url):
    health_url = f"{url}/{HTTPRunDB.get_api_path_prefix()}/healthz"
    timeout = 90
    if not wait_for_server(health_url, timeout):
        raise RuntimeError(f"server did not start after {timeout} sec")


def create_workdir(root_dir="/tmp"):
    return mkdtemp(prefix="mlrun-test-", dir=root_dir)


def start_server(workdir, env_config: dict):
    port = free_port()
    env = environ.copy()
    env["MLRUN_HTTPDB__PORT"] = str(port)
    env["MLRUN_HTTPDB__DSN"] = (
        f"sqlite:///{workdir}/mlrun.sqlite3?check_same_thread=false"
    )
    env["MLRUN_HTTPDB__LOGS_PATH"] = workdir
    env.update(env_config or {})
    cmd = [
        executable,
        "-m",
        "server.api.main",
    ]

    proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE, cwd=project_dir_path)
    url = f"http://localhost:{port}"

    return proc, url


def docker_fixture():
    container_id, workdir = None, None

    def create(env_config=None):
        nonlocal container_id, workdir

        env_config = {} if env_config is None else env_config
        cmd = [
            "docker",
            "build",
            "-f",
            "dockerfiles/mlrun-api/Dockerfile",
            "--tag",
            docker_tag,
            ".",
        ]
        run(cmd, check=True, stdout=PIPE, cwd=project_dir_path)
        workdir = create_workdir(root_dir="/tmp")

        cmd = [
            "docker",
            "run",
            "--detach",
            "--publish",
            "8080",
            # For debugging
            "--volume",
            f"{workdir}:/tmp",
        ]

        env_config.setdefault("MLRUN_HTTPDB__LOGS_PATH", "/tmp")
        for key, value in env_config.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.append(docker_tag)
        out = run(cmd, stdout=PIPE, check=True)
        container_id = out.stdout.decode("utf-8").strip()

        # retrieve container bind port + host
        out = run(["docker", "port", container_id, "8080"], stdout=PIPE, check=True)
        # usually the output is something like b'0.0.0.0:49154\n' but sometimes (in GH actions) it's something like
        # b'0.0.0.0:49154\n:::49154\n' for some reason, so just taking the first line
        host = out.stdout.decode("utf-8").splitlines()[0]

        url = f"http://{host}"
        print(f"api url: {url}")
        check_server_up(url)
        conn = HTTPRunDB(url)
        conn.connect()
        return Server(url, conn, workdir)

    def cleanup():
        if container_id:
            run(["docker", "rm", "--force", container_id], stdout=DEVNULL)
        if workdir:
            rmtree(workdir)

    return create, cleanup


def server_fixture():
    process = None
    workdir = None

    def create(env=None):
        nonlocal process, workdir
        workdir = create_workdir()
        process, url = start_server(workdir, env)
        check_server_up(url)
        conn = HTTPRunDB(url)
        conn.connect()
        return Server(url, conn, workdir)

    def cleanup():
        if process:
            process.terminate()
            stdout = process.stdout.read()
            human_readable_stdout = codecs.escape_decode(stdout)[0].decode("utf-8")
            stderr = process.stderr.read()
            human_readable_stderr = codecs.escape_decode(stderr)[0].decode("utf-8")
            print(f"Stdout from server {human_readable_stdout}")
            print(f"Stderr from server {human_readable_stderr}")
        if workdir:
            rmtree(workdir)

    return create, cleanup


servers = [
    "server",
    "docker",
]


@pytest.fixture(scope="function", params=servers)
def create_server(request):
    if request.param == "server":
        create, cleanup = server_fixture()
    else:
        create, cleanup = docker_fixture()

    try:
        yield create
    finally:
        cleanup()


def test_log(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid, body = "p19", "3920", b"log data"
    proj_obj = mlrun.new_project(prj, save=False)
    db.create_project(proj_obj)

    db.store_run({"metadata": {"name": "run-name"}, "asd": "asd"}, uid, prj)
    db.store_log(uid, prj, body)

    state, data = db.get_log(uid, prj)
    assert data == body, "bad log data"


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="We are developing on Apple Silicon Macs,"
    " which will most likely fail this test due to the qemu being slow,"
    " but should pass on native architecture",
)
def test_api_boot_speed(create_server):
    run_times = 5
    expected_time = 30
    runs = []
    for _ in range(run_times):
        start_time = time.perf_counter()
        create_server()
        end_time = time.perf_counter()
        runs.append(end_time - start_time)
    avg_run_time = sum(runs) / run_times
    assert (
        avg_run_time <= expected_time
    ), "Seems like a performance hit on creating api server"


def test_run(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid = "p18", "3i920"
    proj_obj = mlrun.new_project(prj, save=False)
    db.create_project(proj_obj)

    run_as_dict = RunObject().to_dict()
    run_as_dict["metadata"].update({"name": "run-name", "algorithm": "svm", "C": 3})
    db.store_run(run_as_dict, uid, prj)

    data = db.read_run(uid, prj)
    assert (
        deepdiff.DeepDiff(
            data,
            run_as_dict,
            ignore_order=True,
            exclude_paths={
                "root['status']['start_time']",
                "root['status']['last_update']",
            },
        )
        == {}
    )

    new_c = 4
    updates = {"metadata.C": new_c}
    db.update_run(updates, uid, prj)
    data = db.read_run(uid, prj)
    assert data["metadata"]["C"] == new_c, "update_run"

    db.del_run(uid, prj)


def test_runs(create_server):
    server: Server = create_server()
    db = server.conn
    prj = "p180"
    proj_obj = mlrun.new_project(prj, save=False)
    db.create_project(proj_obj)

    runs = db.list_runs()
    assert not runs, "found runs in new db"
    count = 7

    run_as_dict = RunObject().to_dict()
    for i in range(count):
        uid = f"uid_{i}"
        run_as_dict["metadata"]["name"] = "run-name"
        if i % 2 == 0:
            run_as_dict["status"]["state"] = "completed"
        else:
            run_as_dict["status"]["state"] = "created"
        db.store_run(run_as_dict, uid, prj)

    # retrieve only the last run as it is partitioned by name
    # and since there is no other filter, it will return only the last run
    runs = db.list_runs(project=prj)
    assert len(runs) == 1, "bad number of runs"

    # retrieve all runs
    runs = db.list_runs(
        project=prj,
        start_time_from=datetime.datetime.now() - datetime.timedelta(days=1),
    )
    assert len(runs) == 7, "bad number of runs"

    # retrieve only created runs
    runs = db.list_runs(project=prj, states=["created"])
    assert len(runs) == 3, "bad number of runs"

    # retrieve created and completed runs
    runs = db.list_runs(project=prj, states=["created", "completed"])
    assert len(runs) == 7, "bad number of runs"

    # delete runs in created state
    db.del_runs(project=prj, state="created")

    # delete runs in completed state
    db.del_runs(project=prj, state="completed")

    runs = db.list_runs(project=prj)
    assert not runs, "found runs in after delete"


def test_basic_auth(create_server):
    user, password = "bugs", "bunny"
    env = {
        "MLRUN_HTTPDB__AUTHENTICATION__MODE": "basic",
        "MLRUN_HTTPDB__AUTHENTICATION__BASIC__USERNAME": user,
        "MLRUN_HTTPDB__AUTHENTICATION__BASIC__PASSWORD": password,
    }
    server: Server = create_server(env)

    db: HTTPRunDB = server.conn

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        db.list_runs()

    db.user = user
    db.password = password
    db.list_runs()


def test_bearer_auth(create_server):
    token = "banana"
    env = {
        "MLRUN_HTTPDB__AUTHENTICATION__MODE": "bearer",
        "MLRUN_HTTPDB__AUTHENTICATION__BEARER__TOKEN": token,
    }
    server: Server = create_server(env)

    db: HTTPRunDB = server.conn

    with pytest.raises(mlrun.errors.MLRunUnauthorizedError):
        db.list_runs()

    db.token_provider = StaticTokenProvider(token)
    db.list_runs()


def test_client_id_auth(requests_mock: requests_mock_package.Mocker, monkeypatch):
    """
    Test the httpdb behavior when using a client-id OAuth token. Test verifies that:
    - Token is retrieved successfully, and kept in the httpdb class.
    - Token is added as Bearer token when issuing API calls to BE.
    - Token is refreshed when its expiry time is nearing.
    - Some error flows when token cannot be retrieved - such as that token is still used while it hasn't expired.
    """

    token_url = "https://mock/token_endpoint/protocol/openid-connect/token"
    test_env = {
        "MLRUN_AUTH_TOKEN_ENDPOINT": token_url,
        "MLRUN_AUTH_CLIENT_ID": "some-client-id",
        "MLRUN_AUTH_CLIENT_SECRET": "some-client-secret",
    }

    mlrun.mlconf.auth_with_client_id.enabled = True
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    expected_token = "my-cool-token"
    # Set a 4-second expiry, so a refresh will happen in 2 seconds
    requests_mock.post(
        token_url, json={"access_token": expected_token, "expires_in": 4}
    )

    db_url = "http://mock-server:1919"
    db = HTTPRunDB(db_url)
    db.connect()
    token = db.token_provider.get_token()
    assert token == expected_token
    assert len(requests_mock.request_history) == 1

    time.sleep(1)
    token = db.token_provider.get_token()
    assert token == expected_token
    # verify no additional calls were made (too early)
    assert len(requests_mock.request_history) == 1

    time.sleep(1.5)
    expected_token = "my-other-cool-token"
    requests_mock.post(
        token_url, json={"access_token": expected_token, "expires_in": 3}
    )
    token = db.token_provider.get_token()
    assert token == expected_token

    # Check that httpdb attaches the token to API calls as Authorization header.
    # Using trigger-migrations since it needs no payload and returns nothing, so easy to simulate.
    requests_mock.post(f"{db_url}/api/v1/operations/migrations", status_code=200)
    db.trigger_migrations()

    expected_auth = f"Bearer {expected_token}"
    last_request = requests_mock.last_request
    assert last_request.headers["Authorization"] == expected_auth

    # Check flow where we fail token retrieval while token is still active (not expired).
    requests_mock.reset_mock()
    requests_mock.post(token_url, status_code=401)

    time.sleep(2)
    db.trigger_migrations()

    request_history = requests_mock.request_history
    # We expect 2 calls - one for the token (which failed but didn't fail the flow) and one for the actual api call.
    assert len(request_history) == 2
    # The token should still be the previous token, since it was not refreshed but it's not expired yet.
    assert request_history[-1].headers["Authorization"] == expected_auth

    # Now let the token expire, and verify commands still go out, only without auth
    time.sleep(2)
    requests_mock.reset_mock()

    db.trigger_migrations()
    assert len(requests_mock.request_history) == 2
    assert "Authorization" not in requests_mock.last_request.headers
    assert db.token_provider.token is None


def _generate_runtime(name) -> mlrun.runtimes.KubejobRuntime:
    runtime = mlrun.runtimes.KubejobRuntime()
    runtime.metadata.name = name
    return runtime


def test_set_get_function(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn
    name = "test"
    project = "project"
    func = _generate_runtime(name)
    func.set_label("new", "label")
    tag = uuid4().hex
    proj_obj = mlrun.new_project(project, save=False)
    db.create_project(proj_obj)

    db.store_function(func.to_dict(), name, project, tag=tag)
    db_func = db.get_function(name, project, tag=tag)

    assert db_func["metadata"]["name"] == name
    assert db_func["metadata"]["labels"]["new"] == "label"


def test_list_functions(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    proj = "p4"
    proj_obj = mlrun.new_project(proj, save=False)
    db.create_project(proj_obj)

    count = 5
    for i in range(count):
        name = f"func{i}"
        func = {"fid": i}
        tag = uuid4().hex
        db.store_function(func, name, proj, tag=tag)
    proj_p7 = "p7"
    proj_p7_obj = mlrun.new_project(proj_p7, save=False)
    db.create_project(proj_p7_obj)

    db.store_function({}, "f2", proj_p7, tag=uuid4().hex)

    functions = db.list_functions(project=proj)
    for function in functions:
        assert function["metadata"]["tag"] is not None
    assert len(functions) == count, "bad list"


@pytest.mark.parametrize(
    "server_version,client_version,compatible",
    [
        # Unstable client or server, not parsing, and assuming compatibility
        ("unstable", "unstable", True),
        ("0.5.3", "unstable", True),
        ("unstable", "0.6.1", True),
        # Server and client versions are not the same but compatible
        ("0.5.3", "0.5.1", True),
        ("0.6.0-rc1", "0.6.1", True),
        ("0.6.0-rc1", "0.5.4", True),
        ("0.6.3", "0.4.8", True),
        ("1.3.0", "1.1.0", True),
        # Majors on the server and client versions are not the same
        ("1.0.0", "0.5.0", False),
        ("0.5.0", "1.0.0", False),
        ("2.0.0", "1.3.0", False),
        ("2.0.0", "1.9.0", False),
        # Server version much higher than client
        ("1.3.0", "1.0.0", False),
        ("1.9.0", "1.3.0", False),
        # Client version higher than server, not supported
        ("1.3.0", "1.9.0", False),
        ("1.3.0", "1.4.0", False),
        # Server or client version is unstable, assuming compatibility
        ("0.7.1", "0.0.0+unstable", True),
        ("0.0.0+unstable", "0.7.1", True),
        # feature branch
        ("0.7.1", "0.0.0+feature-branch", True),
        ("0.7.1-rc1", "0.0.0+feature-branch", True),
        ("0.7.1-rc1+feature-branch", "0.0.0+feature-branch", True),
        ("0.7.1", "0.7.1+feature-branch", True),
        ("0.7.1-rc1", "0.7.1+feature-branch", True),
        ("0.7.1-rc1+feature-branch", "0.7.1+feature-branch", True),
    ],
)
def test_version_compatibility_validation(server_version, client_version, compatible):
    assert compatible == HTTPRunDB._validate_version_compatibility(
        server_version, client_version
    )


def _create_feature_set(name):
    return {
        "kind": "FeatureSet",
        "metadata": {
            "name": name,
            "labels": {"owner": "saarc", "group": "dev"},
            "tag": "latest",
        },
        "spec": {
            "entities": [
                {
                    "name": "ticker",
                    "value_type": "str",
                    "labels": {"type": "prod"},
                    "extra_field": 100,
                }
            ],
            "features": [
                {"name": "time", "value_type": "datetime", "extra_field": "value1"},
                {"name": "bid", "value_type": "float"},
                {"name": "ask", "value_type": "time"},
            ],
        },
        "status": {
            "state": "created",
            "stats": {
                "time": {
                    "count": "8",
                    "unique": "7",
                    "top": "2016-05-25 13:30:00.222222",
                }
            },
            "preview": [
                [
                    "time",
                    "bid",
                    "ask",
                ],
                [
                    "2016-05-25 13:30:00.222222",
                    7.3,
                    "10:30:00.222222",
                ],
                [
                    "2016-05-24 13:30:00.222222",
                    7.3,
                    "11:30:00.222222",
                ],
                [
                    "2016-05-23 13:30:00.222222",
                    4.7,
                    "13:20:00.222222",
                ],
                [
                    "2016-05-22 13:30:00.222222",
                    5.2,
                    "13:15:00.222222",
                ],
                [
                    "2016-05-21 13:30:00.222222",
                    5,
                    "18:30:00.222222",
                ],
                [
                    "2016-05-20 13:30:00.222222",
                    4.6,
                    "09:30:00.222222",
                ],
                [
                    "2016-05-19 13:30:00.222222",
                    5.6,
                    "08:30:00.222222",
                ],
                [
                    "2016-05-24 13:30:00.222222",
                    5.6,
                    "13:30:00.222222",
                ],
            ],
        },
        "some_other_field": "blabla",
    }


def test_feature_sets(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project = "newproj"
    proj_obj = mlrun.new_project(project, save=False)
    db.create_project(proj_obj)

    count = 5
    for i in range(count):
        name = f"fs_{i}"
        feature_set = _create_feature_set(name)
        db.create_feature_set(feature_set, project=project, versioned=True)

    # Test store_feature_set, which allows updates as well as inserts
    db.store_feature_set(feature_set, name=name, project=project, versioned=True)

    feature_set_update = {
        "spec": {
            "features": [{"name": "looks", "value_type": "str", "description": "good"}],
        }
    }

    # additive mode means add the feature to the features-list
    db.patch_feature_set(
        name, feature_set_update, project, tag="latest", patch_mode="additive"
    )
    feature_sets = db.list_feature_sets(project=project)
    assert len(feature_sets) == count

    feature_sets = db.list_feature_sets(
        project=project,
        partition_by="name",
        rows_per_partition=1,
        partition_sort_by="updated",
        partition_order="desc",
    )
    assert len(feature_sets) == count
    assert all([feature_set.status.stats for feature_set in feature_sets])
    assert all([feature_set.status.preview for feature_set in feature_sets])

    feature_set = db.get_feature_set(name, project)
    assert len(feature_set.spec.features) == 4

    # test minimal feature set format
    feature_sets = db.list_feature_sets(
        project=project,
        partition_by="name",
        rows_per_partition=1,
        partition_sort_by="updated",
        partition_order="desc",
        format_=mlrun.common.formatters.FeatureSetFormat.minimal,
    )
    assert len(feature_sets) == count
    assert not any([feature_set.status.stats for feature_set in feature_sets])
    assert not any([feature_set.status.preview for feature_set in feature_sets])
    assert all([feature_set.status.state for feature_set in feature_sets])

    # Create a feature-set that has no labels
    name = "feature_set_no_labels"
    feature_set_without_labels = _create_feature_set(name)
    feature_set_without_labels["metadata"].pop("labels")
    # Use project name in the feature-set (don't provide it to API)
    feature_set_without_labels["metadata"]["project"] = project
    db.store_feature_set(feature_set_without_labels)
    feature_set_update = {
        "spec": {"entities": [{"name": "nothing", "value_type": "bool"}]},
        "metadata": {"labels": {"label1": "value1", "label2": "value2"}},
    }
    db.patch_feature_set(name, feature_set_update, project)
    feature_set = db.get_feature_set(name, project)
    assert len(feature_set.metadata.labels) == 2, "Labels didn't get updated"

    features = db.list_features(project, "time")
    # The feature-set with different labels also counts here
    assert len(features) == count + 1
    # Only count, since we modified the entity of the last feature-set - other name, no labels
    entities = db.list_entities(project, "ticker")
    assert len(entities) == count
    entities = db.list_entities(project, labels=["type"])
    assert len(entities) == count
    entities = db.list_entities(project, labels=["type=prod"])
    assert len(entities) == count


def test_remove_labels_from_feature_set(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project = "newproj"
    proj_obj = mlrun.new_project(project, save=False)
    db.create_project(proj_obj)

    feature_set = _create_feature_set("feature-set-test")
    db.create_feature_set(feature_set, project=project, versioned=True)

    feature_sets = db.list_feature_sets(project=project)
    assert len(feature_sets) == 1, "bad number of feature sets"
    assert len(feature_sets[0].metadata.labels) == 2, "bad number of labels"
    assert (
        feature_sets[0].metadata.labels == feature_set["metadata"]["labels"]
    ), "labels were not set correctly"

    feature_set = feature_sets[0]
    feature_set.metadata.labels = {}
    db.store_feature_set(feature_set.to_dict(), project=project)
    feature_sets = db.list_feature_sets(project=project, tag="latest")
    assert feature_sets[0].metadata.labels == {}, "labels were not removed correctly"


def _create_feature_vector(name):
    return {
        "kind": "FeatureVector",
        "metadata": {
            "name": name,
            "labels": {"owner": "nobody", "group": "dev"},
            "tag": "latest",
        },
        "spec": {
            "features": [
                "feature_set.*",
                "feature_set.something",
                "feature_set.just_a_feature",
            ],
            "description": "just a bunch of features",
        },
        "status": {"state": "created"},
    }


def test_tagging_artifacts(create_server):
    _, db = _configure_run_db_server(create_server)

    tag = "tag"
    add_tag = "new-tag"
    proj_obj, logged_artifact = _generate_project_and_artifact(tag=tag)

    db.tag_artifacts(logged_artifact, proj_obj.name, tag_name=add_tag)

    _assert_artifacts(db, proj_obj.name, tag, 1)
    _assert_artifacts(db, proj_obj.name, add_tag, 1)


def test_replacing_artifact_tags(create_server):
    _, db = _configure_run_db_server(create_server)

    tag = "tag"
    new_tag = "new-tag"
    proj_obj, logged_artifact = _generate_project_and_artifact(tag=tag)

    _assert_artifacts(db, proj_obj.name, tag, 1)

    db.tag_artifacts(logged_artifact, proj_obj.name, tag_name=new_tag, replace=True)

    _assert_artifacts(db, proj_obj.name, tag, 0)
    _assert_artifacts(db, proj_obj.name, new_tag, 1)


def test_delete_artifact_tags(create_server):
    _, db = _configure_run_db_server(create_server)

    tag = "tag"
    new_tag = "new-tag"
    proj_obj, logged_artifact = _generate_project_and_artifact(tag=tag)

    _assert_artifacts(db, proj_obj.name, tag, 1)

    db.tag_artifacts(logged_artifact, proj_obj.name, tag_name=new_tag)

    _assert_artifacts(db, proj_obj.name, new_tag, 1)
    _assert_artifacts(db, proj_obj.name, tag, 1)

    db.delete_artifacts_tags(logged_artifact, proj_obj.name, tag_name=tag)

    _assert_artifacts(db, proj_obj.name, new_tag, 1)
    _assert_artifacts(db, proj_obj.name, tag, 0)


def test_add_tag_and_delete_untagged_artifacts(create_server):
    _, db = _configure_run_db_server(create_server)
    project_name = "artifact-project"
    project = mlrun.new_project(project_name)

    # create 4 artifacts that are basically the same, but with different auto-generated trees to create different uids.
    # only the last one will get the latest tag
    artifact_key = "artifact_key"
    # add a different db_key to simulate artifact created by a run
    artifact_db_key = f"{project_name}-{artifact_key}"
    num_artifacts = 4
    for i in range(num_artifacts):
        project.log_artifact(
            artifact_key,
            body=b"some data",
            db_key=artifact_db_key,
        )

    # list all artifacts
    artifacts = db.list_artifacts(project=project_name)
    assert len(artifacts) == num_artifacts
    artifact_tags = [artifact["metadata"].get("tag") for artifact in artifacts]
    assert artifact_tags.count("latest") == 1
    assert artifact_tags.count(None) == num_artifacts - 1

    # find untagged artifacts and add a new tag to them
    untagged_artifacts = [
        artifact
        for artifact in artifacts
        if "tag" not in artifact["metadata"] or artifact["metadata"]["tag"] is None
    ]
    new_tags = []
    for idx, untagged_artifact in enumerate(untagged_artifacts):
        new_tag = f"new-tag-{idx}"
        new_tags.append(new_tag)
        db.tag_artifacts(untagged_artifact, project_name, tag_name=new_tag)

    # verify the artifacts were tagged
    artifact_tags = db.list_artifact_tags(project=project_name)
    assert len(artifact_tags) == num_artifacts

    artifacts = db.list_artifacts(project=project_name)
    artifact_tags = [artifact["metadata"].get("tag") for artifact in artifacts]
    assert len(artifact_tags) == num_artifacts
    assert artifact_tags.count("latest") == 1
    assert artifact_tags.count(None) == 0

    # delete a single artifact with a new tag
    db.del_artifact(
        key=artifact_db_key,
        tag=new_tags[0],
        project=project_name,
    )

    # list all artifacts
    artifacts = db.list_artifacts(project=project_name)
    assert len(artifacts) == num_artifacts - 1

    # delete the rest of the artifacts with 'delete_artifacts'
    artifacts_to_delete = [
        artifact for artifact in artifacts if artifact["metadata"]["tag"] != "latest"
    ]
    for artifact_to_delete in artifacts_to_delete:
        db.del_artifacts(
            name=artifact_db_key,
            tag=artifact_to_delete["metadata"]["tag"],
            project=project_name,
        )

    # verify only the latest remained
    artifacts = db.list_artifacts(project=project_name)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["tag"] == "latest"


def _generate_project_and_artifact(project: str = "newproj", tag: str = None):
    proj_obj = mlrun.new_project(project)

    logged_artifact = proj_obj.log_artifact(
        "my-artifact",
        body=b"some data",
        tag=tag,
    )
    return proj_obj, logged_artifact


def _assert_artifacts(db, project: str, tag: str, expected_count: int):
    artifacts = db.list_artifacts(project=project, tag=tag)
    assert (
        len(artifacts) == expected_count
    ), "bad list results - wrong number of artifacts"


def _configure_run_db_server(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn
    mlrun.mlconf.dbpath = server.url
    mlrun.db._run_db = db
    mlrun.db._last_db_url = server.url

    return server, db


def test_feature_vectors(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project = "newproj"
    proj_obj = mlrun.new_project(project, save=False)
    db.create_project(proj_obj)

    count = 5
    for i in range(count):
        name = f"fs_{i}"
        feature_vector = _create_feature_vector(name)
        db.create_feature_vector(feature_vector, project=project, versioned=True)

    # Test store_feature_set, which allows updates as well as inserts
    db.store_feature_vector(feature_vector, project=project)

    feature_vector_update = {"spec": {"features": ["bla.asd", "blu.asd"]}}

    # additive mode means add the feature to the features-list
    db.patch_feature_vector(
        name,
        feature_vector_update,
        project,
        tag="latest",
        patch_mode=mlrun.common.schemas.PatchMode.additive,
    )
    feature_vectors = db.list_feature_vectors(project=project)
    assert len(feature_vectors) == count, "bad list results - wrong number of members"

    feature_vectors = db.list_feature_vectors(
        project=project,
        partition_by="name",
        rows_per_partition=1,
        partition_sort_by="updated",
        partition_order="desc",
    )
    assert len(feature_vectors) == count, "bad list results - wrong number of members"

    feature_vector = db.get_feature_vector(name, project)
    assert (
        len(feature_vector.spec.features) == 5
    ), "Features didn't get updated properly"

    # Create a feature-vector that has no labels
    name = "feature_vector_no_labels"
    feature_vector_without_labels = _create_feature_vector(name)
    feature_vector_without_labels["metadata"].pop("labels")
    # Use project name in the feature-set (don't provide it to API)
    feature_vector_without_labels["metadata"]["project"] = project
    db.store_feature_vector(feature_vector_without_labels)

    # Perform a replace (vs. additive as done earlier) - now should only have 2 features
    db.patch_feature_vector(
        name,
        feature_vector_update,
        project,
        patch_mode=mlrun.common.schemas.PatchMode.replace,
    )
    feature_vector = db.get_feature_vector(name, project)
    assert (
        len(feature_vector.spec.features) == 2
    ), "Features didn't get updated properly"


def test_project_sql_db_roundtrip(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project_name = "project-name"
    description = "project description"
    goals = "project goals"
    desired_state = mlrun.common.schemas.ProjectState.archived
    params = {"param_key": "param value"}
    artifact_path = "/tmp"
    conda = "conda"
    source = "source"
    subpath = "subpath"
    origin_url = "origin_url"
    labels = {"key": "value"}
    annotations = {"annotation-key": "annotation-value"}
    project_metadata = mlrun.projects.project.ProjectMetadata(
        project_name,
        labels=labels,
        annotations=annotations,
    )
    project_spec = mlrun.projects.project.ProjectSpec(
        description,
        params,
        artifact_path=artifact_path,
        conda=conda,
        source=source,
        subpath=subpath,
        origin_url=origin_url,
        goals=goals,
        desired_state=desired_state,
    )
    project = mlrun.projects.project.MlrunProject(
        metadata=project_metadata, spec=project_spec
    )
    function_name = "trainer-function"
    function = mlrun.new_function(function_name, project_name)
    project.set_function(function, function_name)
    workflow_name = "workflow-name"
    workflow_file_path = Path(tests_root_directory) / "rundb" / "workflow.py"
    project.set_workflow(workflow_name, str(workflow_file_path))
    artifact_dict = {
        "key": "raw-data",
        "kind": "",
        "iter": 0,
        "tree": "latest",
        "target_path": "https://raw.githubusercontent.com/mlrun/demos/master/customer-churn-prediction/WA_Fn-UseC_-Telc"
        "o-Customer-Churn.csv",
        "db_key": "raw-data",
    }
    project.artifacts = [artifact_dict]
    created_project = db.create_project(project)
    _assert_projects(project, created_project)
    stored_project = db.store_project(project_name, project)
    _assert_projects(project, stored_project)
    patched_project = db.patch_project(project_name, {})
    _assert_projects(project, patched_project)
    get_project = db.get_project(project_name)
    _assert_projects(project, get_project)
    list_projects = db.list_projects(format_=mlrun.common.formatters.ProjectFormat.full)
    _assert_projects(project, list_projects[0])


def _assert_projects(expected_project, project):
    assert (
        deepdiff.DeepDiff(
            expected_project.to_dict(),
            project.to_dict(),
            ignore_order=True,
            exclude_paths={
                "root['metadata']['created']",
                "root['spec']['desired_state']",
                "root['status']",
            },
        )
        == {}
    )
    assert expected_project.spec.desired_state == project.spec.desired_state
    assert expected_project.spec.desired_state == project.status.state


@pytest.mark.parametrize(
    "alert_name_in_config, alert_name_as_func_param",
    [
        (None, None),
        (None, ""),
        ("", None),
        ("", ""),
    ],
)
def test_store_alert_config_missing_alert_name(
    alert_name_in_config, alert_name_as_func_param, create_server
):
    server: Server = create_server()
    db: HTTPRunDB = server.conn
    alert_data = mlrun.alerts.alert.AlertConfig(name=alert_name_in_config, project=None)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="Alert name must be provided"
    ):
        db.store_alert_config(
            alert_name=alert_name_as_func_param,
            alert_data=alert_data,
        )
