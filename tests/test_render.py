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

import pytest

import mlrun
import mlrun.lists
import mlrun.render
from tests.conftest import results, rundb_path

assets_path = pathlib.Path(__file__).parent / "assets"
function_path = str(assets_path / "log_function.py")


def get_db():
    return mlrun.get_run_db(rundb_path)


@pytest.mark.parametrize(
    "generate_artifact_hash_mode, expected_target_paths",
    [
        (
            False,
            [
                f"{results}/log-function-log-dataset/0/feature_1.csv",
                f"{results}/log-function-log-dataset/0/feature_2.csv",
            ],
        ),
        (
            True,
            [
                f"{results}/6154c46f1a6fffb0b6b716882279d7e09ecb6b8a.csv",
                f"{results}/c88c2dc877a6595cb2eb834449aac6e2789d301c.csv",
            ],
        ),
    ],
)
def test_list_runs(rundb_mock, generate_artifact_hash_mode, expected_target_paths):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = (
        generate_artifact_hash_mode
    )
    func = mlrun.code_to_function(
        filename=function_path, kind="job", handler="log_dataset"
    )
    run = func.run(local=True, out_path=str(results))

    # Verify target path in enriched run list
    runs = mlrun.lists.RunList([run.to_dict()])
    html = runs.show(display=False)
    for expected_target_path in expected_target_paths:
        expected_link, _ = mlrun.render.link_to_ipython(expected_target_path)
        assert expected_link in html

    runs = rundb_mock.list_runs()
    assert runs, "empty runs result"

    # Verify store URI in not-enriched runs
    html = runs.show(display=False)
    dataset_0_uri = list(runs[0]["status"]["artifact_uris"].values())[0]
    assert dataset_0_uri
    assert dataset_0_uri in html


# FIXME: this test was counting on the fact it's running after some test (I think test_httpdb) which leaves runs and
#  artifacts in the `results` dir, it should generate its own stuff, skipping for now
@pytest.mark.skip("FIX_ME")
def test_list_artifacts():
    db = get_db()
    artifacts = db.list_artifacts()
    assert artifacts, "empty artifacts result"

    html = artifacts.show(display=False)

    with open(f"{results}/artifacts.html", "w") as fp:
        fp.write(html)
