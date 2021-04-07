def vault_func(context, secrets: list):
    """Validate that given secrets exists

    :param context: the MLRun context
    :param secrets: name of the secrets that we want to look at
    """
    context.logger.info("running function")
    for sec_name in secrets:
        sec_value = context.get_secret(sec_name)
        context.logger.info("Secret name: {}, value: {}".format(sec_name, sec_value))

    return True
