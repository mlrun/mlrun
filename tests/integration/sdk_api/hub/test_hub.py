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

import random
import shutil
import types
from pathlib import Path

import pytest

import mlrun
import mlrun.common.schemas
import tests.integration.sdk_api.base
from mlrun.utils import normalize_name


class TestHub(tests.integration.sdk_api.base.TestMLRunIntegration):
    @staticmethod
    def _assert_source_lists_match(expected_response):
        response = mlrun.get_run_db().list_hub_sources()
        for i in range(len(expected_response)):
            assert expected_response[i].source.diff(response[i].source) == {}

    def test_hub(self):
        db = mlrun.get_run_db()

        response = mlrun.get_run_db().list_hub_sources()
        # make sure that there is only the default source
        assert len(response) == 1
        default_source = response[0]

        new_source = mlrun.common.schemas.IndexedHubSource(
            source=mlrun.common.schemas.HubSource(
                metadata=mlrun.common.schemas.HubObjectMetadata(
                    name="source-1", description="a private source"
                ),
                spec=mlrun.common.schemas.HubSourceSpec(
                    path="/local/path/to/source", channel="development"
                ),
            )
        )
        db.create_hub_source(new_source)
        new_source.index = 1
        self._assert_source_lists_match([new_source, default_source])

        new_source_2 = mlrun.common.schemas.IndexedHubSource(
            index=1,
            source=mlrun.common.schemas.HubSource(
                metadata=mlrun.common.schemas.HubObjectMetadata(
                    name="source-2", description="2nd private source"
                ),
                spec=mlrun.common.schemas.HubSourceSpec(
                    path="/local/path/to/source", channel="prod"
                ),
            ),
        )

        db.create_hub_source(new_source_2)
        new_source.index = 2
        self._assert_source_lists_match([new_source_2, new_source, default_source])

        new_source.index = 1
        db.store_hub_source(new_source.source.metadata.name, new_source)
        new_source_2.index = 2
        self._assert_source_lists_match([new_source, new_source_2, default_source])

        db.delete_hub_source("source-1")
        new_source_2.index = 1
        self._assert_source_lists_match([new_source_2, default_source])

    def test_import_function_from_hub(self):
        hub_prefix = "hub://"
        source_name = mlrun.mlconf.hub.default_source.name
        db = mlrun.get_run_db()
        catalog = db.get_hub_catalog(source_name)
        item = random.choice(catalog.catalog)
        tag = item.metadata.tag
        name = item.metadata.name
        # plain option
        fn = mlrun.import_function(hub_prefix + name)
        assert fn.metadata.name == name
        # source option
        fn = mlrun.import_function(hub_prefix + source_name + "/" + name)
        assert fn.metadata.name == name
        # source and tag option
        fn = mlrun.import_function(hub_prefix + source_name + "/" + name + ":" + tag)
        assert fn.metadata.name == name
        # tag option
        fn = mlrun.import_function(hub_prefix + name + ":" + tag)
        assert fn.metadata.name == name
        # not existed option
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            mlrun.import_function(hub_prefix + source_name + "-not" + "/" + name)

    def test_get_hub_module(self):
        hub_prefix = "hub://"
        source_name = mlrun.mlconf.hub.default_source.name
        db = mlrun.get_run_db()
        modules_catalog = db.get_hub_catalog(
            source_name, object_type=mlrun.common.schemas.hub.HubSourceType.modules
        )
        item = random.choice(modules_catalog.catalog)
        tag = item.metadata.tag
        name = item.metadata.name
        # plain option
        hub_module = mlrun.get_hub_module(hub_prefix + name, download_files=False)
        assert normalize_name(hub_module.name) == name
        # source option
        hub_module = mlrun.get_hub_module(
            hub_prefix + source_name + "/" + name, download_files=False
        )
        assert normalize_name(hub_module.name) == name
        # tag option
        hub_module = mlrun.get_hub_module(
            hub_prefix + source_name + "/" + name + ":" + tag, download_files=False
        )
        assert normalize_name(hub_module.name) == name
        if tag != "latest":
            assert hub_module.version == tag
        # not existed option
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            mlrun.get_hub_module(
                hub_prefix + source_name + "-not" + "/" + name, download_files=False
            )

    def test_import_module_from_hub(self):
        hub_prefix = "hub://"
        source_name = mlrun.mlconf.hub.default_source.name
        db = mlrun.get_run_db()
        modules_catalog = db.get_hub_catalog(
            source_name, object_type=mlrun.common.schemas.hub.HubSourceType.modules
        )
        item = random.choice(modules_catalog.catalog)
        name = item.metadata.name

        # import_module
        # create temp dir in cwd
        Path.cwd().joinpath("temp").mkdir(exist_ok=True)
        mod = mlrun.import_module(hub_prefix + name, local_path="./temp")
        assert isinstance(mod, types.ModuleType)
        # delete the temp dir
        shutil.rmtree("temp")

        # get_hub_module and module
        Path.cwd().joinpath("temp").mkdir(exist_ok=True)
        hub_module = mlrun.get_hub_module(hub_prefix + name, download_files=False)
        with pytest.raises(FileNotFoundError):  # didn't download files first
            hub_module.module()
        hub_module.download_module_files("./temp")
        mod = hub_module.module()
        assert isinstance(mod, types.ModuleType)
        # delete the temp dir
        shutil.rmtree("temp")

        # local_path doesn't exist
        with pytest.raises(ValueError):
            mlrun.import_module(hub_prefix + name, local_path="./temp")
