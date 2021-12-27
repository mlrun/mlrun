def secret_test_function(context, secrets: list = []):
    """Validate that given secrets exists

    :param context: the MLRun context
    :param secrets: name of the secrets that we want to look at
    """
    context.logger.info("running function")
    for sec_name in secrets:
        sec_value = context.get_secret(sec_name)
        context.logger.info("Secret: {} ==> {}".format(sec_name, sec_value))
        context.log_result(sec_name, sec_value)
    return True
