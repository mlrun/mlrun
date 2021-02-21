def store_path_to_spark(path):
    if path.startswith("v3io:///"):
        path = "v3io:" + path[len("v3io:/") :]
    return path
