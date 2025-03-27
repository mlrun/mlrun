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
import pytest

import mlrun.errors

import framework.db.sqldb.helpers
import framework.db.sqldb.models
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestHelpers(TestDatabaseBase):
    @pytest.mark.parametrize(
        "labels",
        [
            ("my-str"),
            ([]),
            (None),
            ({"a": [{"b": "c"}]}),
            ({1: "a"}),
            ({"a" * 256: "b"}),
            ({"a": "b" * 256}),
        ],
    )
    def test_update_labels_invalid(self, labels):
        obj = framework.db.sqldb.models.ArtifactV2()
        obj.labels = []
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            framework.db.sqldb.helpers.update_labels(obj, labels)
