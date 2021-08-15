import mlrun

mlrun.mlconf.feature_store.flush_interval = None


def test_func(context, p1):
    return p1 + 1
