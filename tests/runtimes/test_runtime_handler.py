import unittest.mock

from mlrun.runtimes import get_runtime_handler, RuntimeKinds


def test_list_kubejob_resources(k8s_helper_mock):
    pods = _create_kubejob_pod_mocks()
    k8s_helper_mock.list_pods.return_value = pods
    runtime_handler = get_runtime_handler(RuntimeKinds.job)
    resources = runtime_handler.list_resources()
    k8s_helper_mock.list_pods.assert_called_once_with(
        k8s_helper_mock.resolve_namespace(),
        selector=runtime_handler._get_default_label_selector(),
    )
    _assert_resources(resources, expected_pods=pods)


def test_list_daskjob_resources(k8s_helper_mock):
    pods = _create_daskjob_pod_mocks()
    services = _create_daskjob_service_mocks()
    k8s_helper_mock.list_pods.return_value = pods
    k8s_helper_mock.v1api.list_namespaced_service.return_value = services
    runtime_handler = get_runtime_handler(RuntimeKinds.dask)
    resources = runtime_handler.list_resources()
    k8s_helper_mock.list_pods.assert_called_once_with(
        k8s_helper_mock.resolve_namespace(),
        selector=runtime_handler._get_default_label_selector(),
    )
    k8s_helper_mock.v1api.list_namespaced_service.assert_called_once_with(
        k8s_helper_mock.resolve_namespace(),
        label_selector=runtime_handler._get_default_label_selector(),
    )
    _assert_resources(resources, expected_pods=pods, expected_services=services)


def _assert_resources(
    resources, expected_crds=None, expected_pods=None, expected_services=None
):
    if expected_crds is None:
        expected_crds = []
    if expected_pods is None:
        expected_pods = []
    if expected_services is None:
        expected_services = []
    assert resources['crd_resources'] == expected_crds
    assert len(resources['pod_resources']) == len(expected_pods)
    for index, pod in enumerate(expected_pods):
        pod_dict = pod.to_dict()
        assert resources['pod_resources'][index]['name'] == pod_dict['metadata']['name']
        assert (
            resources['pod_resources'][index]['labels']
            == pod_dict['metadata']['labels']
        )
        assert resources['pod_resources'][index]['status'] == pod_dict['status']
    assert len(resources['service_resources']) == len(expected_services.items)
    for index, service in enumerate(expected_services.items):
        assert resources['service_resources'][index]['name'] == service.metadata.name
        assert (
            resources['service_resources'][index]['labels'] == service.metadata.labels
        )


def _create_daskjob_service_mocks():
    service_mock = unittest.mock.Mock()
    service_mock.metadata.name.return_value = 'mlrun-mydask-d7656bc1-0'
    service_mock.metadata.labels.return_value = {
        'app': 'dask',
        'dask.org/cluster-name': 'mlrun-mydask-d7656bc1-0',
        'dask.org/component': 'scheduler',
        'mlrun/class': 'dask',
        'mlrun/function': 'mydask',
        'mlrun/project': 'default',
        'mlrun/scrape_metrics': 'False',
        'mlrun/tag': 'latest',
        'user': 'root',
    }
    services_mock = unittest.mock.Mock()
    services_mock.items = [service_mock]
    return services_mock


def _create_kubejob_pod_mocks():
    pod_dict = {
        'metadata': {
            'name': 'my-training-j7dtf',
            'labels': {
                'mlrun/class': 'job',
                'mlrun/function': 'my-trainer',
                'mlrun/name': 'my-training',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': 'bba96b8313b640cd9143d7513000c47c',
            },
        },
        'status': {
            'conditions': [
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:23+00:00',
                    'message': None,
                    'reason': 'PodCompleted',
                    'status': 'True',
                    'type': 'Initialized',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:47+00:00',
                    'message': None,
                    'reason': 'PodCompleted',
                    'status': 'False',
                    'type': 'Ready',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:47+00:00',
                    'message': None,
                    'reason': 'PodCompleted',
                    'status': 'False',
                    'type': 'ContainersReady',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:23+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled',
                },
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://c00c36dc9a702508c76b6074f2c2fa3e569daaf13f5a72931804da04a6e96987',
                    'image': 'docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.com:80/mlrun/func-defaul'
                    't-my-trainer-latest:latest',
                    'image_id': 'docker-pullable://docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.com'
                    ':80/mlrun/func-default-my-trainer-latest@sha256:d23c93a997fa5ab89d899bf1bf1cb97fa506'
                    '97a74c61927c1df3266340076efc',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None,
                    },
                    'name': 'base',
                    'ready': False,
                    'restart_count': 0,
                    'state': {
                        'running': None,
                        'terminated': {
                            'container_id': 'docker://c00c36dc9a702508c76b6074f2c2fa3e569daaf13f5a72931804da04a6e96987',
                            'exit_code': 0,
                            'finished_at': '2020-08-17T18:08:47+00:00',
                            'message': None,
                            'reason': 'Completed',
                            'signal': None,
                            'started_at': '2020-08-17T18:08:42+00:00',
                        },
                        'waiting': None,
                    },
                }
            ],
            'host_ip': '172.31.6.138',
            'init_container_statuses': None,
            'message': None,
            'nominated_node_name': None,
            'phase': 'Succeeded',
            'pod_ip': '10.200.0.48',
            'qos_class': 'BestEffort',
            'reason': None,
            'start_time': '2020-08-17T18:08:23+00:00',
        },
    }
    pod_mock = unittest.mock.Mock()
    pod_mock.to_dict.return_value = pod_dict
    return [pod_mock]


def _create_daskjob_pod_mocks():
    scheduler_pod_dict = {
        'metadata': {
            'name': 'mlrun-mydask-d7656bc1-0n4z9z',
            'labels': {
                'app': 'dask',
                'dask.org/cluster-name': 'mlrun-mydask-d7656bc1-0',
                'dask.org/component': 'scheduler',
                'mlrun/class': 'dask',
                'mlrun/function': 'mydask',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'user': 'root',
            },
        },
        'status': {
            'conditions': [
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:35:15+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Initialized',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:20+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Ready',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:20+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'ContainersReady',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:35:15+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled',
                },
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://c24d68d4c71d8f61bed56aaf424dc7ffb86a9f59d7710afaa1e28f47bd466a5e',
                    'image': 'mlrun/ml-models:0.5.1',
                    'image_id': 'docker-pullable://mlrun/ml-models@sha256:07cc9e991dc603dbe100f4eae93d20f3ce53af1e6d4af'
                    '5191dbc66e6dfbce85b',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None,
                    },
                    'name': 'base',
                    'ready': True,
                    'restart_count': 0,
                    'state': {
                        'running': {'started_at': '2020-08-18T00:36:19+00:00'},
                        'terminated': None,
                        'waiting': None,
                    },
                }
            ],
            'host_ip': '172.31.6.138',
            'init_container_statuses': None,
            'message': None,
            'nominated_node_name': None,
            'phase': 'Running',
            'pod_ip': '10.200.0.48',
            'qos_class': 'BestEffort',
            'reason': None,
            'start_time': '2020-08-18T00:35:15+00:00',
        },
    }
    worker_pod_dict = {
        'metadata': {
            'name': 'mlrun-mydask-d7656bc1-0pqbnc',
            'labels': {
                'app': 'dask',
                'dask.org/cluster-name': 'mlrun-mydask-d7656bc1-0',
                'dask.org/component': 'worker',
                'mlrun/class': 'dask',
                'mlrun/function': 'mydask',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'user': 'root',
            },
        },
        'status': {
            'conditions': [
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:21+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Initialized',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:24+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Ready',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:24+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'ContainersReady',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T00:36:21+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled',
                },
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://18f75b15b9fbf0ed9136d9ec7f14cf1d62dbfa078877f89847fa346fe09ff574',
                    'image': 'mlrun/ml-models:0.5.1',
                    'image_id': 'docker-pullable://mlrun/ml-models@sha256:07cc9e991dc603dbe100f4eae93d20f3ce53af1e6d4af'
                    '5191dbc66e6dfbce85b',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None,
                    },
                    'name': 'base',
                    'ready': True,
                    'restart_count': 0,
                    'state': {
                        'running': {'started_at': '2020-08-18T00:36:23+00:00'},
                        'terminated': None,
                        'waiting': None,
                    },
                }
            ],
            'host_ip': '172.31.6.138',
            'init_container_statuses': None,
            'message': None,
            'nominated_node_name': None,
            'phase': 'Running',
            'pod_ip': '10.200.0.51',
            'qos_class': 'BestEffort',
            'reason': None,
            'start_time': '2020-08-18T00:36:21+00:00',
        },
    }
    scheduler_pod_mock = unittest.mock.Mock()
    scheduler_pod_mock.to_dict.return_value = scheduler_pod_dict
    worker_pod_mock = unittest.mock.Mock()
    worker_pod_mock.to_dict.return_value = worker_pod_dict
    return [scheduler_pod_mock, worker_pod_mock]
