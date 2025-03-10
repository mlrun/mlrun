from mlrun.serving import Model


class DummyModel(Model):
    execution_mechanism = "naive"
