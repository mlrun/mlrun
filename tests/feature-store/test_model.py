from mlrun.featurestore.model import FeatureSet, FeatureVector, Entity, Feature, ValueType


def test_feature_set():
    myset = FeatureSet("test1", entities=[Entity("key")])
    myset['f1'] = Feature(ValueType.INT64, description='my f1')

    assert list(myset.spec.entities.keys()) == ['key'], 'index wasnt set'
    assert list(myset.spec.features.keys()) == ['f1'], 'feature wasnt set'


