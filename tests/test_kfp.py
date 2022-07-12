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

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import yaml

import mlrun.kfpops
from mlrun import new_function, new_task
from mlrun.artifacts import ChartArtifact
from mlrun.utils import logger

model_body = "abc is 123"
results_body = "<b> Some HTML <b>"
tests_dir = Path(__file__).absolute().parent


def my_job(context, p1=1, p2="a-string"):

    # access input metadata, values, files, and secrets (passwords)
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}")
    print(f"accesskey = {context.get_secret('ACCESS_KEY')}")
    input_file = context.get_input(
        str(tests_dir) + "/assets/test_kfp_input_file.txt"
    ).get()
    print(f"file\n{input_file}\n")

    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)

    # log various types of artifacts (file, web page, table), will be
    # versioned and visible in the UI
    context.log_artifact("model", body=model_body, local_path="model.txt")
    context.log_artifact("results", local_path="results.html", body=results_body)

    # create a chart output (will show in the pipelines UI)
    chart = ChartArtifact("chart")
    chart.header = ["Epoch", "Accuracy", "Loss"]
    for i in range(1, 8):
        chart.add_row([i, i / 20 + 0.75, 0.30 - i / 20])
    context.log_artifact(chart)

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "postTestScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(
        raw_data, columns=["first_name", "last_name", "age", "postTestScore"]
    )
    context.log_dataset("mydf", df=df)


@pytest.fixture
def kfp_dirs(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        meta_dir = Path(tmpdir) / "meta"
        artifacts_dir = Path(tmpdir) / "artifacts"
        output_dir = Path(tmpdir) / "output"
        for path in [meta_dir, artifacts_dir, output_dir]:
            os.mkdir(path)
        logger.info(
            "Created temp paths for kfp test",
            meta_dir=meta_dir,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
        )
        monkeypatch.setattr(mlrun.kfpops, "KFPMETA_DIR", str(meta_dir))
        monkeypatch.setattr(mlrun.kfpops, "KFP_ARTIFACTS_DIR", str(artifacts_dir))
        yield (str(meta_dir), str(artifacts_dir), str(output_dir))


def test_kfp_function_run(kfp_dirs):
    meta_dir, artifacts_dir, output_dir = kfp_dirs
    p1 = 5
    expected_accuracy = 2 * p1
    expected_loss = 3 * p1
    task = _generate_task(p1, output_dir)
    result = new_function(kfp=True).run(task, handler=my_job)
    _assert_meta_dir(meta_dir, expected_accuracy, expected_loss)
    _assert_artifacts_dir(artifacts_dir, expected_accuracy, expected_loss)
    _assert_output_dir(output_dir, result.metadata.name)
    assert result.output("accuracy") == expected_accuracy
    assert result.output("loss") == expected_loss
    assert result.status.state == "completed"


def test_kfp_function_run_with_hyper_params(kfp_dirs):
    meta_dir, artifacts_dir, output_dir = kfp_dirs
    p1 = [1, 2, 3]
    task = _generate_task(p1, output_dir)
    task.with_hyper_params({"p1": p1}, selector="min.loss")
    best_iteration = 1  # loss is 3 * p1, so min loss will be when p1=1
    expected_accuracy = 2 * p1[best_iteration - 1]
    expected_loss = 3 * p1[best_iteration - 1]
    result = new_function(kfp=True).run(task, handler=my_job)
    _assert_meta_dir(meta_dir, expected_accuracy, expected_loss, best_iteration)
    _assert_artifacts_dir(artifacts_dir, expected_accuracy, expected_loss)
    _assert_output_dir(output_dir, result.metadata.name, iterations=len(p1))
    assert result.output("accuracy") == expected_accuracy
    assert result.output("loss") == expected_loss
    assert result.status.state == "completed"


def _assert_output_dir(output_dir, name, iterations=1):
    output_prefix = f"{output_dir}/{name}/"
    for iteration in range(1, iterations):
        _assert_iteration_output_dir_files(output_prefix, iteration)
    if iterations > 1:
        iteration_results_file = output_prefix + "0/iteration_results.csv"
        with open(iteration_results_file) as file:
            count = 0
            for row in csv.DictReader(file):
                print(yaml.dump(row))
                count += 1
        assert count == 3, "didnt see expected iterations file output"


def _assert_iteration_output_dir_files(output_dir, iteration):
    def file_path(key):
        return output_dir + f"{iteration}/{key}"

    with open(file_path("model.txt")) as model_file:
        contents = model_file.read()
        assert contents == model_body
    with open(file_path("results.html")) as results_file:
        contents = results_file.read()
        assert contents == results_body
    assert os.path.exists(file_path("chart.html"))
    assert os.path.exists(file_path("mydf.parquet"))


def _assert_artifacts_dir(artifacts_dir, expected_accuracy, expected_loss):
    with open(artifacts_dir + "/accuracy") as accuracy_file:
        accuracy = accuracy_file.read()
        assert str(expected_accuracy) == accuracy
    with open(artifacts_dir + "/loss") as loss_file:
        loss = loss_file.read()
        assert str(expected_loss) == loss


def _assert_meta_dir(meta_dir, expected_accuracy, expected_loss, best_iteration=None):
    _assert_metrics_file(meta_dir, expected_accuracy, expected_loss, best_iteration)
    _assert_ui_metadata_file_existence(meta_dir)


def _assert_ui_metadata_file_existence(meta_dir):
    assert os.path.exists(meta_dir + "/mlpipeline-ui-metadata.json")


def _assert_metrics_file(
    meta_dir, expected_accuracy, expected_loss, best_iteration=None
):
    expected_data = {
        "metrics": [
            {"name": "accuracy", "numberValue": expected_accuracy},
            {"name": "loss", "numberValue": expected_loss},
        ]
    }
    if best_iteration is not None:
        expected_data["metrics"].insert(
            0, {"name": "best_iteration", "numberValue": best_iteration}
        )
    with open(meta_dir + "/mlpipeline-metrics.json") as metrics_file:
        data = json.load(metrics_file)
        assert data == expected_data


def _generate_task(p1, out_path):
    return new_task(
        params={"p1": p1},
        out_path=out_path,
        outputs=["accuracy", "loss"],
    ).set_label("tests", "kfp")
