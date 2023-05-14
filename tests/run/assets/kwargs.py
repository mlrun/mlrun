def func(context, x, **kwargs):
    context.logger.info(x)
    context.logger.info(kwargs)
    if not kwargs:
        raise Exception("kwargs is empty")
    return kwargs
