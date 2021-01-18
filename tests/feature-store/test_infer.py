from mlrun.feature_store.infer import infer_from_df, InferOptions
from tests.conftest import tests_root_directory
import pandas as pd
import mlrun.feature_store as fs

this_dir = f"{tests_root_directory}/feature-store/"

expected_schema = [
    {"name": "bad", "value_type": "int"},
    {"name": "department", "value_type": "str"},
    {"name": "room", "value_type": "int"},
    {"name": "hr", "value_type": "float"},
    {"name": "hr_is_error", "value_type": "bool"},
    {"name": "rr", "value_type": "int"},
    {"name": "rr_is_error", "value_type": "bool"},
    {"name": "spo2", "value_type": "int"},
    {"name": "spo2_is_error", "value_type": "bool"},
    {"name": "movements", "value_type": "float"},
    {"name": "movements_is_error", "value_type": "bool"},
    {"name": "turn_count", "value_type": "float"},
    {"name": "turn_count_is_error", "value_type": "bool"},
    {"name": "is_in_bed", "value_type": "int"},
    {"name": "is_in_bed_is_error", "value_type": "bool"},
    {"name": "timestamp", "value_type": "str"},
]


def test_infer_from_df():
    key = "patient_id"
    df = pd.read_csv(this_dir + "testdata.csv")
    featureset = fs.FeatureSet("testdata", entities=[fs.Entity(key)])
    infer_from_df(df, featureset, options=InferOptions.all())
    # print(featureset.to_yaml())
    assert (
        featureset.spec.features.to_dict() == expected_schema
    ), "did not infer schema properly"

    preview = featureset.status.preview
    # by default preview should be 20 lines + 1 for headers
    assert len(preview) == 21, "unexpected num of preview lines"
    assert len(preview[0]) == df.shape[1], "unexpected num of header columns"
    assert len(preview[1]) == df.shape[1], "unexpected num of value columns"

    features = sorted(featureset.spec.features.keys())
    stats = sorted(featureset.status.stats.keys())
    stats.remove(key)
    assert features == stats, "didnt infer stats for all features"

    stat_columns = list(featureset.status.stats["movements"].keys())
    assert stat_columns == [
        "count",
        "mean",
        "std",
        "min",
        "max",
        "hist",
    ], "wrong stats result"
