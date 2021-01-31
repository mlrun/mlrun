import pandas as pd
import mlrun

from mlrun.artifacts.model import ModelArtifact, update_model, get_model
from mlrun.features import Feature
from tests.conftest import results

results_dir = f"{results}/artifacts/"

raw_data = {
    "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
    "age": [42, 52, 36, 24, 73],
    "testScore": [25, 94, 57, 62, 70],
}

expected_inputs = [
    {"name": "last_name", "value_type": "str"},
    {"name": "first_name", "value_type": "str"},
    {"name": "age", "value_type": "int"},
]
expected_outputs = [{"name": "testScore", "value_type": "int"}]


def test_infer():
    model = ModelArtifact("my-model")
    df = pd.DataFrame(raw_data, columns=["last_name", "first_name", "age", "testScore"])
    model.infer_from_df(df, ["testScore"])
    assert model.inputs.to_dict() == expected_inputs, "unexpected model inputs"
    assert model.outputs.to_dict() == expected_outputs, "unexpected model outputs"
    assert list(model.feature_stats.keys()) == [
        "last_name",
        "first_name",
        "age",
        "testScore",
    ], "wrong stat keys"


def test_model_update():
    model = ModelArtifact("my-model", model_file="a.pkl")
    target_path = results_dir + "model/"

    project = mlrun.new_project("test-proj")
    artifact = project.log_artifact(model, upload=False, artifact_path=target_path)

    artifact_uri = f"store://artifacts/{artifact.project}/{artifact.db_key}"
    updated_model_spec = update_model(
        artifact_uri,
        parameters={"a": 1},
        metrics={"b": 2},
        inputs=[Feature(name="f1")],
        outputs=[Feature(name="f2")],
        feature_vector="vec",
        feature_weights=[1, 2],
        key_prefix="test-",
        labels={"lbl": "tst"},
        write_spec_copy=False,
    )
    print(updated_model_spec.to_yaml())

    model_path, model, extra_dataitems = get_model(artifact_uri)
    # print(model_spec.to_yaml())
    assert model_path.endswith(f"model/{model.model_file}"), "illegal model path"
    assert model.parameters == {"a": 1}, "wrong parameters"
    assert model.metrics == {"test-b": 2}, "wrong metrics"

    assert model.inputs[0].name == "f1", "wrong inputs"
    assert model.outputs[0].name == "f2", "wrong outputs"

    assert model.feature_vector == "vec", "wrong feature_vector"
    assert model.feature_weights == [1, 2], "wrong feature_weights"
    assert model.labels == {"lbl": "tst"}, "wrong labels"
