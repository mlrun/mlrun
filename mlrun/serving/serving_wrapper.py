# serving runtime hooks, used in empty serving functions
from mlrun.runtimes import nuclio_init_hook


def init_context(context):
    nuclio_init_hook(context, globals(), "serving_v2")


def handler(context, event):
    return context.mlrun_handler(context, event)
