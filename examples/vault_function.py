def vault_func(context, name, value=None):
    """Validate that a given secret exists

    :param context the MLRun context
    :param name name of the secret
    :param value expected value of the secret
    """
    context.logger.info("running function")
    sec_value = context.get_secret(name)

    context.logger.info("Secret value: {}".format(sec_value))

    if sec_value is None or (value and sec_value != value):
        return False

    return True


def embedded_secret_func(context, value):
    context.logger.info("running function")

    sec_value = {{ Secret.password }}
    return value == sec_value
