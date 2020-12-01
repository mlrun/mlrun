def build_endpoint_key(project: str, function: str, model: str, version: str):
    return f"{project}_{function}_{model}_{version}"
