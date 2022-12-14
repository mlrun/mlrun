import notexisting  # noqa: F401


def handler(context):
    context.log_result("accuracy", 16)
