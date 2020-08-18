import unittest.mock

from mlrun.runtimes import get_runtime_handler, RuntimeKinds


def test_list_kubejob_resources(k8s_helper_mock):
    pods = _mock_list_kubejob_pods(k8s_helper_mock)
    runtime_handler = get_runtime_handler(RuntimeKinds.job)
    _assert_runtime_handler_list_resources(
        k8s_helper_mock, runtime_handler, expected_pods=pods
    )


def test_list_daskjob_resources(k8s_helper_mock):
    pods = _mock_list_daskjob_pods(k8s_helper_mock)
    services = _create_daskjob_service_mocks(k8s_helper_mock)
    runtime_handler = get_runtime_handler(RuntimeKinds.dask)
    _assert_runtime_handler_list_resources(
        k8s_helper_mock, runtime_handler, expected_pods=pods, expected_services=services
    )


def test_list_mpijob_resources(k8s_helper_mock):
    crds = _mock_list_mpijob_crds(k8s_helper_mock)
    k8s_helper_mock.list_pods.return_value = []
    runtime_handler = get_runtime_handler(RuntimeKinds.mpijob)
    _assert_runtime_handler_list_resources(
        k8s_helper_mock, runtime_handler, expected_crds=crds
    )


def test_list_sparkjob_resources(k8s_helper_mock):
    crds = _mock_list_sparkjob_crds(k8s_helper_mock)
    pods = _mock_list_daskjob_pods(k8s_helper_mock)
    runtime_handler = get_runtime_handler(RuntimeKinds.spark)
    _assert_runtime_handler_list_resources(
        k8s_helper_mock, runtime_handler, expected_crds=crds, expected_pods=pods
    )


def _assert_runtime_handler_list_resources(
    k8s_helper_mock,
    runtime_handler,
    expected_crds=None,
    expected_pods=None,
    expected_services=None,
):
    resources = runtime_handler.list_resources()
    crd_group, crd_version, crd_plural = runtime_handler._get_crd_info()
    k8s_helper_mock.list_pods.assert_called_once_with(
        k8s_helper_mock.resolve_namespace(),
        selector=runtime_handler._get_default_label_selector(),
    )
    if expected_crds:
        k8s_helper_mock.crdapi.list_namespaced_custom_object.assert_called_once_with(
            crd_group,
            crd_version,
            k8s_helper_mock.resolve_namespace(),
            crd_plural,
            label_selector=runtime_handler._get_default_label_selector(),
        )
    if expected_services:
        k8s_helper_mock.v1api.list_namespaced_service.assert_called_once_with(
            k8s_helper_mock.resolve_namespace(),
            label_selector=runtime_handler._get_default_label_selector(),
        )
    _assert_list_resources_response(
        resources,
        expected_crds=expected_crds,
        expected_pods=expected_pods,
        expected_services=expected_services,
    )


def _assert_list_resources_response(
    resources, expected_crds=None, expected_pods=None, expected_services=None
):
    if expected_crds is None:
        expected_crds = []
    else:
        expected_crds = expected_crds['items']
    if expected_pods is None:
        expected_pods = []
    if expected_services is None:
        expected_services = []
    else:
        expected_services = expected_services.items
    assert len(resources['crd_resources']) == len(expected_crds)
    for index, crd in enumerate(expected_crds):
        assert resources['crd_resources'][index]['name'] == crd['metadata']['name']
        assert resources['crd_resources'][index]['labels'] == crd['metadata']['labels']
        assert resources['crd_resources'][index]['status'] == crd['status']
    assert len(resources['pod_resources']) == len(expected_pods)
    for index, pod in enumerate(expected_pods):
        pod_dict = pod.to_dict()
        assert resources['pod_resources'][index]['name'] == pod_dict['metadata']['name']
        assert (
            resources['pod_resources'][index]['labels']
            == pod_dict['metadata']['labels']
        )
        assert resources['pod_resources'][index]['status'] == pod_dict['status']
    if expected_services:
        assert len(resources['service_resources']) == len(expected_services)
        for index, service in enumerate(expected_services):
            assert (
                resources['service_resources'][index]['name'] == service.metadata.name
            )
            assert (
                resources['service_resources'][index]['labels']
                == service.metadata.labels
            )


def _mock_list_pods(k8s_helper_mock, pod_dicts):
    pod_mocks = []
    for pod_dict in pod_dicts:
        pod_mock = unittest.mock.Mock()
        pod_mock.to_dict.return_value = pod_dict
        pod_mocks.append(pod_mock)
    k8s_helper_mock.list_pods.return_value = pod_mocks
    return pod_mocks


def _mock_list_crds(k8s_helper_mock, crd_dicts):
    crds = {
        'items': crd_dicts,
    }
    k8s_helper_mock.crdapi.list_namespaced_custom_object.return_value = crds
    return crds


def _mock_list_services(k8s_helper_mock, service_dicts):
    service_mocks = []
    for service_dict in service_dicts:
        service_mock = unittest.mock.Mock()
        service_mock.metadata.name.return_value = service_dict['metadata']['name']
        service_mock.metadata.labels.return_value = service_dict['metadata']['labels']
        service_mocks.append(service_mock)
    services_mock = unittest.mock.Mock()
    services_mock.items = service_mocks
    k8s_helper_mock.v1api.list_namespaced_service.return_value = services_mock
    return services_mock


def _mock_list_kubejob_pods(k8s_helper_mock):
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
    return _mock_list_pods(k8s_helper_mock, [pod_dict])


def _create_daskjob_service_mocks(k8s_helper_mock):
    service_dict = {
        'metadata': {
            'name': 'mlrun-mydask-d7656bc1-0',
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
    }
    return _mock_list_services(k8s_helper_mock, [service_dict])


def _mock_list_daskjob_pods(k8s_helper_mock):
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
    return _mock_list_pods(k8s_helper_mock, [scheduler_pod_dict, worker_pod_dict])


def _mock_list_mpijob_crds(k8s_helper_mock):
    crd_dict = {
        'metadata': {
            'name': 'train-eaf63df8',
            'labels': {
                'mlrun/class': 'mpijob',
                'mlrun/function': 'trainer',
                'mlrun/name': 'train',
                'mlrun/project': 'cat-and-dog-servers',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': '9401e4b27f004c6ba750d3e936f1fccb',
            },
        },
        'status': {
            'completionTime': '2020-08-18T01:23:54Z',
            'conditions': [
                {
                    'lastTransitionTime': '2020-08-18T01:21:15Z',
                    'lastUpdateTime': '2020-08-18T01:21:15Z',
                    'message': 'MPIJob default-tenant/train-eaf63df8 is created.',
                    'reason': 'MPIJobCreated',
                    'status': 'True',
                    'type': 'Created',
                },
                {
                    'lastTransitionTime': '2020-08-18T01:21:23Z',
                    'lastUpdateTime': '2020-08-18T01:21:23Z',
                    'message': 'MPIJob default-tenant/train-eaf63df8 is running.',
                    'reason': 'MPIJobRunning',
                    'status': 'False',
                    'type': 'Running',
                },
                {
                    'lastTransitionTime': '2020-08-18T01:23:54Z',
                    'lastUpdateTime': '2020-08-18T01:23:54Z',
                    'message': 'MPIJob default-tenant/train-eaf63df8 successfully completed.',
                    'reason': 'MPIJobSucceeded',
                    'status': 'True',
                    'type': 'Succeeded',
                },
            ],
            'replicaStatuses': {'Launcher': {'succeeded': 1}, 'Worker': {}},
            'startTime': '2020-08-18T01:21:15Z',
        },
    }
    return _mock_list_crds(k8s_helper_mock, [crd_dict])


def _mock_list_sparkjob_crds(k8s_helper_mock):
    crd_dict = {
        'metadata': {
            'name': 'my-spark-jdbc-2ea432f1',
            'labels': {
                'mlrun/class': 'spark',
                'mlrun/function': 'my-spark-jdbc',
                'mlrun/name': 'my-spark-jdbc',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': 'b532ba206a1649da9925d340d6f97f7a',
            },
        },
        'status': {
            'applicationState': {'state': 'RUNNING'},
            'driverInfo': {
                'podName': 'my-spark-jdbc-2ea432f1-driver',
                'webUIAddress': '10.197.111.54:0',
                'webUIPort': 4040,
                'webUIServiceName': 'my-spark-jdbc-2ea432f1-ui-svc',
            },
            'executionAttempts': 2,
            'executorState': {'my-spark-jdbc-2ea432f1-1597760338437-exec-1': 'RUNNING'},
            'sparkApplicationId': 'spark-12f88a73cb544ce298deba34947226a4',
            'submissionAttempts': 1,
            'submissionID': '44343f6b-42ca-41d4-b01a-66052cc5c919',
            'submissionTime': '2020-08-18T14:19:16Z',
            'terminationTime': None,
        },
    }
    return _mock_list_crds(k8s_helper_mock, [crd_dict])


def _mock_list_sparkjob_pods(k8s_helper_mock):
    executor_pod_dict = {
        'metadata': {
            'name': 'my-spark-jdbc-2ea432f1-1597760338437-exec-1',
            'labels': {
                'mlrun/class': 'spark',
                'mlrun/function': 'my-spark-jdbc',
                'mlrun/job': 'my-spark-jdbc-2ea432f1',
                'mlrun/name': 'my-spark-jdbc',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': 'b532ba206a1649da9925d340d6f97f7a',
                'spark-app-selector': 'spark-12f88a73cb544ce298deba34947226a4',
                'spark-exec-id': '1',
                'spark-role': 'executor',
                'sparkoperator.k8s.io/app-name': 'my-spark-jdbc-2ea432f1',
                'sparkoperator.k8s.io/launched-by-spark-operator': 'true',
                'sparkoperator.k8s.io/submission-id': '44343f6b-42ca-41d4-b01a-66052cc5c919',
            },
        },
        'status': {
            'conditions': [
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:25+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Initialized',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:28+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Ready',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:28+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'ContainersReady',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:25+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled',
                },
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://de6c8574b113b1200bae56918e77b4f8f344f18741d8f53cdb5eab5c55f6c16a',
                    'image': 'iguazio/spark-app:2.10_b59_20200813105414',
                    'image_id': 'docker://sha256:251e43e69e8449dc45883ad4e5d3cf785068fa86852335d69e56b605c6bd03'
                    '0b',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None,
                    },
                    'name': 'executor',
                    'ready': True,
                    'restart_count': 0,
                    'state': {
                        'running': {'started_at': '2020-08-18T14:19:28+00:00'},
                        'terminated': None,
                        'waiting': None,
                    },
                }
            ],
            'host_ip': '172.31.7.224',
            'init_container_statuses': None,
            'message': None,
            'nominated_node_name': None,
            'phase': 'Running',
            'pod_ip': '10.200.0.53',
            'qos_class': 'Burstable',
            'reason': None,
            'start_time': '2020-08-18T14:19:25+00:00',
        },
    }
    driver_pod_dict = {
        'metadata': {
            'name': 'my-spark-jdbc-2ea432f1-driver',
            'labels': {
                'mlrun/class': 'spark',
                'mlrun/function': 'my-spark-jdbc',
                'mlrun/job': 'my-spark-jdbc-2ea432f1',
                'mlrun/name': 'my-spark-jdbc',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': 'b532ba206a1649da9925d340d6f97f7a',
                'spark-app-selector': 'spark-12f88a73cb544ce298deba34947226a4',
                'spark-role': 'driver',
                'sparkoperator.k8s.io/app-name': 'my-spark-jdbc-2ea432f1',
                'sparkoperator.k8s.io/launched-by-spark-operator': 'true',
                'sparkoperator.k8s.io/submission-id': '44343f6b-42ca-41d4-b01a-66052cc5c919',
            },
        },
        'status': {
            'conditions': [
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:08+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Initialized',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:17+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'Ready',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:17+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'ContainersReady',
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-18T14:19:08+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled',
                },
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://916268e7baf76e95fc3a8b79227c4807e4f421004e6674649faaa0540d6cad29',
                    'image': 'iguazio/spark-app:2.10_b59_20200813105414',
                    'image_id': 'docker://sha256:251e43e69e8449dc45883ad4e5d3cf785068fa86852335d69e56b605c6bd030b',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None,
                    },
                    'name': 'spark-kubernetes-driver',
                    'ready': True,
                    'restart_count': 0,
                    'state': {
                        'running': {'started_at': '2020-08-18T14:19:16+00:00'},
                        'terminated': None,
                        'waiting': None,
                    },
                }
            ],
            'host_ip': '172.31.7.224',
            'init_container_statuses': None,
            'message': None,
            'nominated_node_name': None,
            'phase': 'Running',
            'pod_ip': '10.200.0.52',
            'qos_class': 'Burstable',
            'reason': None,
            'start_time': '2020-08-18T14:19:08+00:00',
        },
    }
    return _mock_list_pods(k8s_helper_mock, [executor_pod_dict, driver_pod_dict])
