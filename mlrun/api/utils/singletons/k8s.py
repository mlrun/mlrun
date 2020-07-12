from mlrun.k8s_utils import K8sHelper

# TODO: something nicer
k8s: K8sHelper = None


def get_k8s() -> K8sHelper:
    global k8s
    return k8s


def initialize_k8s():
    global k8s
    try:
        k8s = K8sHelper()
    except Exception:
        pass
