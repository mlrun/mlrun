from unittest.mock import Mock

from mlrun.runtimes import get_runtime_handler, RuntimeKinds
# fixtures for test, aren't used directly so we need to ignore the lint here
from tests.common_fixtures import k8s_helper_mock  # noqa: F401


def test_list_kubejobs_resources(k8s_helper_mock):
    pod_dict = {
        'metadata': {
            'name': 'pod_name',
            'labels': {
                'mlrun/class': 'job',
                'mlrun/function': 'my-trainer',
                'mlrun/name': 'my-training',
                'mlrun/project': 'default',
                'mlrun/scrape_metrics': 'False',
                'mlrun/tag': 'latest',
                'mlrun/uid': 'bba96b8313b640cd9143d7513000c47c'
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
                    'type': 'Initialized'
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:47+00:00',
                    'message': None,
                    'reason': 'PodCompleted',
                    'status': 'False',
                    'type': 'Ready'
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:47+00:00',
                    'message': None,
                    'reason': 'PodCompleted',
                    'status': 'False',
                    'type': 'ContainersReady'
                },
                {
                    'last_probe_time': None,
                    'last_transition_time': '2020-08-17T18:08:23+00:00',
                    'message': None,
                    'reason': None,
                    'status': 'True',
                    'type': 'PodScheduled'
                }
            ],
            'container_statuses': [
                {
                    'container_id': 'docker://c00c36dc9a702508c76b6074f2c2fa3e569daaf13f5a72931804da04a6e96987',
                    'image': 'docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.com:80/mlrun/func-default-my-trainer-latest:latest',
                    'image_id': 'docker-pullable://docker-registry.default-tenant.app.hedingber-210-1.iguazio-cd0.com:80/mlrun/func-default-my-trainer-latest@sha256:d23c93a997fa5ab89d899bf1bf1cb97fa50697a74c61927c1df3266340076efc',
                    'last_state': {
                        'running': None,
                        'terminated': None,
                        'waiting': None
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
                            'started_at': '2020-08-17T18:08:42+00:00'
                        },
                        'waiting': None
                    }
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
            'start_time': '2020-08-17T18:08:23+00:00'
        },
    }
    pod = Mock()
    pod.to_dict.return_value = pod_dict
    pods = [pod]
    k8s_helper_mock.list_pods.return_value = pods
    runtime_handler = get_runtime_handler(RuntimeKinds.job)
    resources = runtime_handler.list_resources()
    assert resources['crd_resources'] == []
    assert len(resources['pod_resources']) == len(pods)
    assert resources['pod_resources'][0]['name'] == pod_dict['metadata']['name']
    assert resources['pod_resources'][0]['labels'] == pod_dict['metadata']['labels']
    assert resources['pod_resources'][0]['status'] == pod_dict['status']
