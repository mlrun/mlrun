import storey

from mlrun.serving import GraphContext


def myhand(x, context=None):
    assert isinstance(context, GraphContext), "didnt get a valid context"
    return x * 2


class Mycls(storey.MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        return event * 2
