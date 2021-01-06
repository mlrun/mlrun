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

import json
import pathlib
from datetime import datetime, timedelta, timezone
from os import makedirs, path, remove, scandir, listdir
from typing import List, Union

import yaml
from dateutil.parser import parse as parse_time

import mlrun.api.schemas
import mlrun.errors
from .base import RunDBError, RunDBInterface
from ..config import config
from ..datastore import store_manager
from ..lists import ArtifactList, RunList
from ..utils import (
    dict_to_json,
    dict_to_yaml,
    get_in,
    logger,
    match_labels,
    match_value,
    match_times,
    update_in,
    fill_function_hash,
    generate_object_uri,
)

run_logs = "runs"
artifacts_dir = "artifacts"
functions_dir = "functions"
schedules_dir = "schedules"


class FileRunDB(RunDBInterface):
    kind = "file"

    def __init__(self, dirpath="", format=".yaml"):
        self.format = format
        self.dirpath = dirpath
        self._datastore = None
        self._subpath = None
        makedirs(self.schedules_dir, exist_ok=True)

    def connect(self, secrets=None):
        sm = store_manager.set(secrets)
        self._datastore, self._subpath = sm.get_or_create_store(self.dirpath)
        return self

    def store_log(self, uid, project="", body=None, append=False):
        filepath = self._filepath(run_logs, project, uid, "") + ".log"
        makedirs(path.dirname(filepath), exist_ok=True)
        mode = "ab" if append else "wb"
        with open(filepath, mode) as fp:
            fp.write(body)
            fp.close()

    def get_log(self, uid, project="", offset=0, size=0):
        filepath = self._filepath(run_logs, project, uid, "") + ".log"
        if pathlib.Path(filepath).is_file():
            with open(filepath, "rb") as fp:
                if offset:
                    fp.seek(offset)
                if not size:
                    size = 2 ** 18
                return "", fp.read(size)
        return "", None

    def _run_path(self, uid, iter):
        if iter:
            return "{}-{}".format(uid, iter)
        return uid

    def store_run(self, struct, uid, project="", iter=0):
        data = self._dumps(struct)
        filepath = (
            self._filepath(run_logs, project, self._run_path(uid, iter), "")
            + self.format
        )
        self._datastore.put(filepath, data)

    def update_run(self, updates: dict, uid, project="", iter=0):
        run = self.read_run(uid, project, iter=iter)
        # TODO: Should we raise if run not found?
        if run and updates:
            for key, val in updates.items():
                update_in(run, key, val)
        self.store_run(run, uid, project, iter=iter)

    def read_run(self, uid, project="", iter=0):
        filepath = (
            self._filepath(run_logs, project, self._run_path(uid, iter), "")
            + self.format
        )
        if not pathlib.Path(filepath).is_file():
            raise mlrun.errors.MLRunNotFoundError(uid)
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_runs(
        self,
        name="",
        uid=None,
        project="",
        labels=None,
        state="",
        sort=True,
        last=1000,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
    ):
        labels = [] if labels is None else labels
        filepath = self._filepath(run_logs, project)
        results = RunList()
        if isinstance(labels, str):
            labels = labels.split(",")
        for run, _ in self._load_list(filepath, "*"):
            if (
                match_value(name, run, "metadata.name")
                and match_labels(get_in(run, "metadata.labels", {}), labels)
                and match_value(state, run, "status.state")
                and match_value(uid, run, "metadata.uid")
                and match_times(
                    start_time_from, start_time_to, run, "status.start_time",
                )
                and match_times(
                    last_update_time_from,
                    last_update_time_to,
                    run,
                    "status.last_update",
                )
                and (iter or get_in(run, "metadata.iteration", 0) == 0)
            ):
                results.append(run)

        if sort or last:
            results.sort(
                key=lambda i: get_in(i, ["status", "start_time"], ""), reverse=True
            )
        if last and len(results) > last:
            return RunList(results[:last])
        return results

    def del_run(self, uid, project="", iter=0):
        filepath = (
            self._filepath(run_logs, project, self._run_path(uid, iter), "")
            + self.format
        )
        self._safe_del(filepath)

    def del_runs(self, name="", project="", labels=None, state="", days_ago=0):

        labels = [] if labels is None else labels
        if not any([name, state, days_ago, labels]):
            raise RunDBError(
                "filter is too wide, select name and/or state and/or days_ago"
            )

        filepath = self._filepath(run_logs, project)
        if isinstance(labels, str):
            labels = labels.split(",")

        if days_ago:
            days_ago = datetime.now() - timedelta(days=days_ago)

        def date_before(run):
            d = get_in(run, "status.start_time", "")
            if not d:
                return False
            return parse_time(d) < days_ago

        for run, p in self._load_list(filepath, "*"):
            if (
                match_value(name, run, "metadata.name")
                and match_labels(get_in(run, "metadata.labels", {}), labels)
                and match_value(state, run, "status.state")
                and (not days_ago or date_before(run))
            ):
                self._safe_del(p)

    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        if "updated" not in artifact:
            artifact["updated"] = datetime.now(timezone.utc).isoformat()
        data = self._dumps(artifact)
        if iter:
            key = "{}-{}".format(iter, key)
        filepath = self._filepath(artifacts_dir, project, key, uid) + self.format
        self._datastore.put(filepath, data)
        filepath = (
            self._filepath(artifacts_dir, project, key, tag or "latest") + self.format
        )
        self._datastore.put(filepath, data)

    def read_artifact(self, key, tag="", iter=None, project=""):
        tag = tag or "latest"
        if iter:
            key = "{}-{}".format(iter, key)
        filepath = self._filepath(artifacts_dir, project, key, tag) + self.format

        if not pathlib.Path(filepath).is_file():
            raise RunDBError(key)
        data = self._datastore.get(filepath)
        return self._loads(data)

    def list_artifacts(
        self, name="", project="", tag="", labels=None, since=None, until=None
    ):
        labels = [] if labels is None else labels
        tag = tag or "latest"
        name = name or ""
        logger.info(f"reading artifacts in {project} name/mask: {name} tag: {tag} ...")
        filepath = self._filepath(artifacts_dir, project, tag=tag)
        results = ArtifactList()
        results.tag = tag
        if isinstance(labels, str):
            labels = labels.split(",")
        if tag == "*":
            mask = "**/*" + name
            if name:
                mask += "*"
        else:
            mask = "**/*"

        time_pred = make_time_pred(since, until)
        for artifact, p in self._load_list(filepath, mask):
            if (name == "" or name in get_in(artifact, "key", "")) and match_labels(
                get_in(artifact, "labels", {}), labels
            ):
                if not time_pred(artifact):
                    continue
                if "artifacts/latest" in p:
                    artifact["tree"] = "latest"
                results.append(artifact)

        return results

    def del_artifact(self, key, tag="", project=""):
        tag = tag or "latest"
        filepath = self._filepath(artifacts_dir, project, key, tag) + self.format
        self._safe_del(filepath)

    def del_artifacts(self, name="", project="", tag="", labels=None):
        labels = [] if labels is None else labels
        tag = tag or "latest"
        filepath = self._filepath(artifacts_dir, project, tag=tag)

        if isinstance(labels, str):
            labels = labels.split(",")
        if tag == "*":
            mask = "**/*" + name
            if name:
                mask += "*"
        else:
            mask = "**/*"

        for artifact, p in self._load_list(filepath, mask):
            if (name == "" or name == get_in(artifact, "key", "")) and match_labels(
                get_in(artifact, "labels", {}), labels
            ):

                self._safe_del(p)

    def store_function(self, function, name, project="", tag="", versioned=False):
        tag = tag or get_in(function, "metadata.tag") or "latest"
        hash_key = fill_function_hash(function, tag)
        update_in(function, "metadata.updated", datetime.now(timezone.utc))
        update_in(function, "metadata.tag", "")
        data = self._dumps(function)
        filepath = (
            path.join(
                self.dirpath,
                "{}/{}/{}/{}".format(
                    functions_dir, project or config.default_project, name, tag
                ),
            )
            + self.format
        )
        self._datastore.put(filepath, data)
        if versioned:

            # the "hash_key" version should not include the status
            function["status"] = None

            # versioned means we want this function to be queryable by its hash key so save another file that the
            # hash key is the file name
            filepath = (
                path.join(
                    self.dirpath,
                    "{}/{}/{}/{}".format(
                        functions_dir, project or config.default_project, name, hash_key
                    ),
                )
                + self.format
            )
            data = self._dumps(function)
            self._datastore.put(filepath, data)
        return hash_key

    def get_function(self, name, project="", tag="", hash_key=""):
        tag = tag or "latest"
        file_name = hash_key or tag
        filepath = (
            path.join(
                self.dirpath,
                "{}/{}/{}/{}".format(
                    functions_dir, project or config.default_project, name, file_name
                ),
            )
            + self.format
        )
        if not pathlib.Path(filepath).is_file():
            function_uri = generate_object_uri(project, name, tag, hash_key)
            raise mlrun.errors.MLRunNotFoundError(f"Function not found {function_uri}")
        data = self._datastore.get(filepath)
        parsed_data = self._loads(data)

        # tag should be filled only when queried by tag
        parsed_data["metadata"]["tag"] = "" if hash_key else tag
        return parsed_data

    def delete_function(self, name: str, project: str = ""):
        raise NotImplementedError()

    def list_functions(self, name=None, project="", tag="", labels=None):
        labels = labels or []
        logger.info(f"reading functions in {project} name/mask: {name} tag: {tag} ...")
        filepath = path.join(
            self.dirpath,
            "{}/{}/".format(functions_dir, project or config.default_project),
        )

        # function name -> tag name -> function dict
        functions_with_tag_filename = {}
        # function name -> hash key -> function dict
        functions_with_hash_key_filename = {}
        # function name -> hash keys set
        function_with_tag_hash_keys = {}
        if isinstance(labels, str):
            labels = labels.split(",")
        mask = "**/*"
        if name:
            filepath = "{}{}/".format(filepath, name)
            mask = "*"
        for func, fullname in self._load_list(filepath, mask):
            if match_labels(get_in(func, "metadata.labels", {}), labels):
                file_name, _ = path.splitext(path.basename(fullname))
                function_name = path.basename(path.dirname(fullname))
                target_dict = functions_with_tag_filename

                tag_name = file_name
                # Heuristic - if tag length if bigger than 20 it's probably a hash key
                if len(tag_name) > 20:  # hash vs tags
                    tag_name = ""
                    target_dict = functions_with_hash_key_filename
                else:
                    function_with_tag_hash_keys.setdefault(function_name, set()).add(
                        func["metadata"]["hash"]
                    )
                update_in(func, "metadata.tag", tag_name)
                target_dict.setdefault(function_name, {})[file_name] = func

        # clean duplicated function e.g. function that was saved both in a hash key filename and tag filename
        for (
            function_name,
            hash_keys_to_function_dict_map,
        ) in functions_with_hash_key_filename.items():
            function_hash_keys_to_remove = []
            for (
                function_hash_key,
                function_dict,
            ) in hash_keys_to_function_dict_map.items():
                if function_hash_key in function_with_tag_hash_keys.get(
                    function_name, set()
                ):
                    function_hash_keys_to_remove.append(function_hash_key)

            for function_hash_key in function_hash_keys_to_remove:
                del hash_keys_to_function_dict_map[function_hash_key]

        results = []
        for functions_map in [
            functions_with_hash_key_filename,
            functions_with_tag_filename,
        ]:
            for function_name, filename_to_function_map in functions_map.items():
                results.extend(filename_to_function_map.values())

        return results

    def _filepath(self, table, project, key="", tag=""):
        if tag == "*":
            tag = ""
        if tag:
            key = "/" + key
        project = project or config.default_project
        return path.join(self.dirpath, "{}/{}/{}{}".format(table, project, tag, key))

    def list_projects(
        self,
        owner: str = None,
        format_: mlrun.api.schemas.Format = mlrun.api.schemas.Format.full,
        labels: List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
    ) -> mlrun.api.schemas.ProjectsOutput:
        if owner or format_ == mlrun.api.schemas.Format.full or labels or state:
            raise NotImplementedError()
        run_dir = path.join(self.dirpath, run_logs)
        if not path.isdir(run_dir):
            return mlrun.api.schemas.ProjectsOutput(projects=[])
        project_names = [
            d for d in listdir(run_dir) if path.isdir(path.join(run_dir, d))
        ]
        return mlrun.api.schemas.ProjectsOutput(projects=project_names)

    def get_project(self, name: str) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def delete_project(
        self,
        name: str,
        deletion_strategy: mlrun.api.schemas.DeletionStrategy = mlrun.api.schemas.DeletionStrategy.default(),
    ):
        raise NotImplementedError()

    def store_project(
        self, name: str, project: mlrun.api.schemas.Project,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def patch_project(
        self,
        name: str,
        project: dict,
        patch_mode: mlrun.api.schemas.PatchMode = mlrun.api.schemas.PatchMode.replace,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    def create_project(
        self, project: mlrun.api.schemas.Project,
    ) -> mlrun.api.schemas.Project:
        raise NotImplementedError()

    @property
    def schedules_dir(self):
        return path.join(self.dirpath, schedules_dir)

    def store_schedule(self, data):
        sched_id = 1 + sum(1 for _ in scandir(self.schedules_dir))
        fname = path.join(self.schedules_dir, "{}{}".format(sched_id, self.format),)
        with open(fname, "w") as out:
            out.write(self._dumps(data))

    def list_schedules(self):
        pattern = "*{}".format(self.format)
        for p in pathlib.Path(self.schedules_dir).glob(pattern):
            with p.open() as fp:
                yield self._loads(fp.read())

        return []

    _encodings = {
        ".yaml": ("to_yaml", dict_to_yaml),
        ".json": ("to_json", dict_to_json),
    }

    def _dumps(self, obj):
        meth_name, enc_fn = self._encodings.get(self.format, (None, None))
        if meth_name is None:
            raise ValueError(f"unsupported format - {self.format}")

        meth = getattr(obj, meth_name, None)
        if meth:
            return meth()

        return enc_fn(obj)

    def _loads(self, data):
        if self.format == ".yaml":
            return yaml.load(data, Loader=yaml.FullLoader)
        else:
            return json.loads(data)

    def _load_list(self, dirpath, mask):
        for p in pathlib.Path(dirpath).glob(mask + self.format):
            if p.is_file():
                if ".ipynb_checkpoints" in p.parts:
                    continue
                data = self._loads(p.read_text())
                if data:
                    yield data, str(p)

    def _safe_del(self, filepath):
        if path.isfile(filepath):
            remove(filepath)
        else:
            raise RunDBError(f"run file is not found or valid ({filepath})")

    def create_feature_set(self, feature_set, project="", versioned=True):
        raise NotImplementedError()

    def get_feature_set(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ):
        raise NotImplementedError()

    def list_features(
        self,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ):
        raise NotImplementedError()

    def list_entities(
        self, project: str, name: str = None, tag: str = None, labels: List[str] = None,
    ):
        raise NotImplementedError()

    def list_feature_sets(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
    ):
        raise NotImplementedError()

    def store_feature_set(
        self, feature_set, name=None, project="", tag=None, uid=None, versioned=True
    ):
        raise NotImplementedError()

    def patch_feature_set(
        self, name, feature_set, project="", tag=None, uid=None, patch_mode="replace",
    ):
        raise NotImplementedError()

    def delete_feature_set(self, name, project=""):
        raise NotImplementedError()

    def create_feature_vector(self, feature_vector, project="", versioned=True) -> dict:
        raise NotImplementedError()

    def get_feature_vector(
        self, name: str, project: str = "", tag: str = None, uid: str = None
    ) -> dict:
        raise NotImplementedError()

    def list_feature_vectors(
        self,
        project: str = "",
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
    ) -> List[dict]:
        raise NotImplementedError()

    def store_feature_vector(
        self, feature_vector, name=None, project="", tag=None, uid=None, versioned=True,
    ):
        raise NotImplementedError()

    def patch_feature_vector(
        self,
        name,
        feature_vector_update: dict,
        project="",
        tag=None,
        uid=None,
        patch_mode="replace",
    ):
        raise NotImplementedError()

    def delete_feature_vector(self, name, project=""):
        raise NotImplementedError()

    def list_pipelines(
        self,
        project: str,
        namespace: str = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        format_: Union[
            str, mlrun.api.schemas.Format
        ] = mlrun.api.schemas.Format.metadata_only,
        page_size: int = None,
    ) -> mlrun.api.schemas.PipelinesOutput:
        raise NotImplementedError()

    def create_project_secrets(
        self,
        project: str,
        provider: str = mlrun.api.schemas.secret.SecretProviderName.vault.value,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def get_project_secrets(
        self,
        project: str,
        token: str,
        provider: str = mlrun.api.schemas.secret.SecretProviderName.vault.value,
        secrets: List[str] = None,
    ) -> mlrun.api.schemas.SecretsData:
        raise NotImplementedError()

    def create_user_secrets(
        self,
        user: str,
        provider: str = mlrun.api.schemas.secret.SecretProviderName.vault.value,
        secrets: dict = None,
    ):
        raise NotImplementedError()

    def list_artifact_tags(self, project=None):
        raise NotImplementedError()


def make_time_pred(since, until):
    if not (since or until):
        return lambda artifact: True

    since = since or datetime.min
    until = until or datetime.max

    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)

    def pred(artifact):
        val = artifact.get("updated")
        if not val:
            return True
        t = parse_time(val).replace(tzinfo=timezone.utc)
        return since <= t <= until

    return pred
