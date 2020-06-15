import pytest
from mlrun.k8s_utils import get_k8s_helper


def test_k8s_get_pod():
    cli = get_k8s_helper()
    pod = cli.get_pod('avi-shell-6d9c467597-89lss', namespace='default-tenant')
    print(pod)


def test_k8s_get_pod():
    cli = get_k8s_helper()
    pod = cli.exec_shell_cmd("echo HELLO", 'v3io_pass')
    print(pod)
