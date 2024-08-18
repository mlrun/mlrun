from mlrun.feature_store.common import parse_feature_string


# ML-7453
def test_parse_feature_string_with_dot_in_feature_set_name():
    feature_set, feature, alias = parse_feature_string(
        "monitoring-llm-server-Qwen-Qwen2-0.5B-latest.*"
    )
    assert feature_set == "monitoring-llm-server-Qwen-Qwen2-0.5B-latest"
    assert feature == "*"
    assert alias is None


def test_parse_feature_string_with_alias():
    feature_set, feature, alias = parse_feature_string("fset.feature as alias")
    assert feature_set == "fset"
    assert feature == "feature"
    assert alias == "alias"
