def store_path_to_spark(path):
    if path.startswith("v3io:///"):
        path = "v3io:" + path[len("v3io:/") :]
    elif path.startswith("s3://"):
        if path.startswith("s3:///"):
            # 's3:///' not supported since mlrun 0.9.0 should use s3://
            from mlrun.errors import MLRunInvalidArgumentError
            valid_path = path.replace('///', '//')
            raise MLRunInvalidArgumentError(
                f"'s3:///' is not supported, try using 's3://' instead.\nUse '{valid_path}'"
            )
        else:
            path = "s3a:" + path[len("s3:") :]
    return path
