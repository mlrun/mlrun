from urllib.parse import urlparse


def store_path_to_spark(path):
    if path.startswith("v3io:///"):
        path = "v3io:" + path[len("v3io:/") :]
    elif path.startswith("s3://"):
        if path.startswith("s3:///"):
            # 's3:///' not supported since mlrun 0.9.0 should use s3:// instead
            from mlrun.errors import MLRunInvalidArgumentError

            valid_path = "s3:" + path[len("s3:/") :]
            raise MLRunInvalidArgumentError(
                f"'s3:///' is not supported, try using 's3://' instead.\nE.g: '{valid_path}'"
            )
        else:
            path = "s3a:" + path[len("s3:") :]
    return path


def parse_kafka_url(url, bootstrap_servers=None):
    bootstrap_servers = bootstrap_servers or []
    url = urlparse(url)
    if url.netloc:
        bootstrap_servers = [url.netloc] + bootstrap_servers
    topic = url.path
    topic = topic.lstrip("/")
    return topic, bootstrap_servers
