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

import codecs
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

import mlrun.errors
import mlrun.projects.project
from mlrun import RunObject
from mlrun.api import schemas
from mlrun.artifacts import Artifact
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
    health_url = f"{url}/api/healthz"
    timeout = 30
    if not wait_for_server(health_url, timeout):
        raise RuntimeError(f"server did not start after {timeout} sec")


def create_workdir(root_dir="/tmp"):
    return mkdtemp(prefix="mlrun-test-", dir=root_dir)


def start_server(workdir, env_config: dict):
    port = free_port()
    env = environ.copy()
    env["MLRUN_httpdb__port"] = str(port)
    env[
        "MLRUN_httpdb__dsn"
    ] = f"sqlite:///{workdir}/mlrun.sqlite3?check_same_thread=false"
    env["MLRUN_httpdb__logs_path"] = workdir
    env.update(env_config or {})
    cmd = [
        executable,
        "-m",
        "mlrun.api.main",
    ]

    proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE, cwd=project_dir_path)
    url = f"http://localhost:{port}"
    check_server_up(url)

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
            "--build-arg",
            "MLRUN_PYTHON_VERSION=3.7.9",
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

        env_config.setdefault("MLRUN_httpdb__logs_path", "/tmp")
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
    db.store_run({"asd": "asd"}, uid, prj)
    db.store_log(uid, prj, body)

    state, data = db.get_log(uid, prj)
    assert data == body, "bad log data"


def test_run(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid = "p18", "3i920"
    run_as_dict = RunObject().to_dict()
    run_as_dict["metadata"].update({"algorithm": "svm", "C": 3})
    db.store_run(run_as_dict, uid, prj)

    data = db.read_run(uid, prj)
    assert data == run_as_dict, "read_run"

    new_c = 4
    updates = {"metadata.C": new_c}
    db.update_run(updates, uid, prj)
    data = db.read_run(uid, prj)
    assert data["metadata"]["C"] == new_c, "update_run"

    db.del_run(uid, prj)


def test_runs(create_server):
    server: Server = create_server()
    db = server.conn

    runs = db.list_runs()
    assert not runs, "found runs in new db"
    count = 7

    prj = "p180"
    run_as_dict = RunObject().to_dict()
    for i in range(count):
        uid = f"uid_{i}"
        db.store_run(run_as_dict, uid, prj)

    runs = db.list_runs(project=prj)
    assert len(runs) == count, "bad number of runs"

    db.del_runs(project=prj, state="created")
    runs = db.list_runs(project=prj)
    assert not runs, "found runs in after delete"


def test_artifact(create_server):
    server: Server = create_server()
    db = server.conn

    prj, uid, key, body = "p7", "u199", "k800", "cucumber"
    artifact = Artifact(key, body)

    db.store_artifact(key, artifact, uid, project=prj)
    # TODO: Need a run file
    # db.del_artifact(key, project=prj)


def test_artifacts(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid, key, body = "p9", "u19", "k802", "tomato"
    artifact = Artifact(key, body, target_path="a.txt")

    db.store_artifact(key, artifact, uid, project=prj)
    db.store_artifact(key, artifact, uid, project=prj, iter=42)
    artifacts = db.list_artifacts(project=prj, tag="*")
    assert len(artifacts) == 2, "bad number of artifacts"
    assert artifacts.objects()[0].key == key, "not a valid artifact object"
    assert artifacts.dataitems()[0].url, "not a valid artifact dataitem"

    artifacts = db.list_artifacts(project=prj, tag="*", iter=0)
    assert len(artifacts) == 1, "bad number of artifacts"

    # Only 1 will be returned since it's only looking for iter 0
    artifacts = db.list_artifacts(project=prj, tag="*", best_iteration=True)
    assert len(artifacts) == 1, "bad number of artifacts"

    db.del_artifacts(project=prj, tag="*")
    artifacts = db.list_artifacts(project=prj, tag="*")
    assert len(artifacts) == 0, "bad number of artifacts after del"


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

    db.token = token
    db.list_runs()


def test_set_get_function(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    func, name, proj = {"x": 1, "y": 2}, "f1", "p2"
    tag = uuid4().hex
    db.store_function(func, name, proj, tag=tag)
    db_func = db.get_function(name, proj, tag=tag)

    # db methods enriches metadata and status
    del db_func["metadata"]
    del db_func["status"]
    assert db_func == func, "wrong func"


def test_list_functions(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    proj = "p4"
    count = 5
    for i in range(count):
        name = f"func{i}"
        func = {"fid": i}
        tag = uuid4().hex
        db.store_function(func, name, proj, tag=tag)
    db.store_function({}, "f2", "p7", tag=uuid4().hex)

    functions = db.list_functions(project=proj)
    for function in functions:
        assert function["metadata"]["tag"] is not None
    assert len(functions) == count, "bad list"


def test_version_compatibility_validation():
    cases = [
        {
            "server_version": "unstable",
            "client_version": "unstable",
            "compatible": True,
        },
        {"server_version": "0.5.3", "client_version": "unstable", "compatible": True},
        {"server_version": "unstable", "client_version": "0.6.1", "compatible": True},
        {"server_version": "0.5.3", "client_version": "0.5.1", "compatible": True},
        {"server_version": "0.6.0-rc1", "client_version": "0.6.1", "compatible": True},
        {
            "server_version": "0.6.0-rc1",
            "client_version": "0.5.4",
            "compatible": False,
        },
        {"server_version": "0.6.3", "client_version": "0.4.8", "compatible": False},
        {"server_version": "1.0.0", "client_version": "0.5.0", "compatible": False},
    ]
    for case in cases:
        if not case["compatible"]:
            with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
                HTTPRunDB._validate_version_compatibility(
                    case["server_version"], case["client_version"]
                )
        else:
            HTTPRunDB._validate_version_compatibility(
                case["server_version"], case["client_version"]
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
        },
        "some_other_field": "blabla",
    }


def test_feature_sets(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project = "newproj"
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
    assert len(feature_sets) == count, "bad list results - wrong number of members"

    feature_sets = db.list_feature_sets(
        project=project,
        partition_by="name",
        rows_per_partition=1,
        partition_sort_by="updated",
        partition_order="desc",
    )
    assert len(feature_sets) == count, "bad list results - wrong number of members"

    feature_set = db.get_feature_set(name, project)
    assert len(feature_set.spec.features) == 4

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


def test_feature_vectors(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project = "newproj"
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
        patch_mode=schemas.PatchMode.additive,
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
        name, feature_vector_update, project, patch_mode=schemas.PatchMode.replace
    )
    feature_vector = db.get_feature_vector(name, project)
    assert (
        len(feature_vector.spec.features) == 2
    ), "Features didn't get updated properly"


def test_project_file_db_roundtrip(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    project_name = "project-name"
    description = "project description"
    goals = "project goals"
    desired_state = mlrun.api.schemas.ProjectState.archived
    params = {"param_key": "param value"}
    artifact_path = "/tmp"
    conda = "conda"
    source = "source"
    subpath = "subpath"
    origin_url = "origin_url"
    labels = {"key": "value"}
    annotations = {"annotation-key": "annotation-value"}
    project_metadata = mlrun.projects.project.ProjectMetadata(
        project_name, labels=labels, annotations=annotations,
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
    project.set_function("hub://describe", "describe")
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
    list_projects = db.list_projects()
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
