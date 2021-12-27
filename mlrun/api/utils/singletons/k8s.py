from mlrun.k8s_utils import K8sHelper, get_k8s_helper


def get_k8s() -> K8sHelper:
    return get_k8s_helper(silent=True)
