import os

import mlrun
from mlrun.secrets import SecretsStore


def get_snowflake_password():
    key = "SNOWFLAKE_PASSWORD"
    snowflake_password = os.getenv(key) or os.getenv(
        SecretsStore.k8s_env_variable_name_for_secret(key)
    )

    if not snowflake_password:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "No password provided. Set password using the SNOWFLAKE_PASSWORD "
            "project secret or environment variable."
        )

    return snowflake_password


def get_spark_options(attributes):
    return {
        "format": "net.snowflake.spark.snowflake",
        "sfURL": attributes.get("url"),
        "sfUser": attributes.get("user"),
        "sfPassword": get_snowflake_password(),
        "sfDatabase": attributes.get("database"),
        "sfSchema": attributes.get("schema"),
        "sfWarehouse": attributes.get("warehouse"),
        "application": "iguazio_platform",
    }
