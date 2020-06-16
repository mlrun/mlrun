import pytest
from mlrun.k8s_utils import get_k8s_helper

v3io_access_key = '<v3io_access_key>'

def test_k8s_get_pod():
    cli = get_k8s_helper()
    pod = cli.get_pod(v3io_access_key, namespace='default-tenant')
    print(pod)


def test_k8s_get_pod():
    cli = get_k8s_helper()
    cmd = "spark-submit"
    pod = cli.exec_shell_cmd(cmd, v3io_access_key)
    print(pod)


def test_validate_access_key():
    cli = get_k8s_helper()
    resp = cli._validate_access_key(v3io_access_key)
    print(resp)