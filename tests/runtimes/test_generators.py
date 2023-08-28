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
#

import pathlib

import pytest

import mlrun.runtimes.generators


@pytest.mark.parametrize(
    "strategy,param_file,expected_generator_class",
    [
        ("list", "hyperparams.csv", mlrun.runtimes.generators.ListGenerator),
        ("grid", "hyperparams.json", mlrun.runtimes.generators.GridGenerator),
        ("random", "hyperparams.json", mlrun.runtimes.generators.RandomGenerator),
        # no strategy, default to list
        ("", "hyperparams.csv", mlrun.runtimes.generators.ListGenerator),
        # no strategy, default to grid
        ("", "hyperparams.json", mlrun.runtimes.generators.GridGenerator),
    ],
)
def test_get_generator(rundb_mock, strategy, param_file, expected_generator_class):
    run_spec = mlrun.model.RunSpec(inputs={"input1": 1})
    run_spec.strategy = strategy
    run_spec.param_file = str(
        pathlib.Path(__file__).absolute().parent / "assets" / param_file
    )
    execution = mlrun.run.MLClientCtx.from_dict(
        mlrun.run.RunObject(spec=run_spec).to_dict(),
        rundb_mock,
        autocommit=False,
        is_api=False,
        store_run=False,
    )
    generator = mlrun.runtimes.generators.get_generator(run_spec, execution, None)
    assert isinstance(
        generator, expected_generator_class
    ), f"unexpected generator type {type(generator)}"

    if strategy == "list":
        assert generator.df.keys().to_list() == ["p1", "p2"]
    elif strategy in ["grid", "random"]:
        assert sorted(list(generator.hyperparams.keys())) == ["p1", "p2"]
