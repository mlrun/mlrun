def store_path_to_spark(path):
    if path.startswith("v3io:///"):
        path = "v3io:" + path[len("v3io:/") :]
    elif path.startswith("s3:///"):
        path = "s3a:" + path[len("s3:/") :]
    return path
