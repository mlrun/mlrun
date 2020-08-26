def my_func(context, p1: int = 1, p2="a-string"):
    """this is a two param function

    :param p1  first param
    :param p2  2nd param
    """
    # access input metadata, values, files, and secrets (passwords)
    print("Run: {} (uid={})".format(context.name, context.uid))
    print("Params: p1={}, p2={}".format(p1, p2))
    context.logger.info("running function")

    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result("accuracy", p1 * 2)
    context.log_result("loss", p1 * 3)
    context.set_label("framework", "sklearn")
