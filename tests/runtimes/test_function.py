from deepdiff import DeepDiff

import mlrun


def test_load_function_from_remote_source_git():
    fn = mlrun.new_function("test-func", kind="nuclio")
    fn.from_remote_source("git://github.com/org/repo#my-branch",
                          handler="path/inside/repo#main:handler",
                          credentials={"GIT_PASSWORD": "my-access-token"})

    assert fn.spec.base_spec == {
        'apiVersion': 'nuclio.io/v1',
        'kind': 'Function',
        'metadata': {'name': 'notebook', 'labels': {}, 'annotations': {}},
        'spec': {
            'runtime': 'python:3.7',
            'handler': 'main:handler',
            'env': [],
            'volumes': [],
            'build': {
                'commands': [],
                'noBaseImagesPull': True,
                'path': 'https://github.com/org/repo',
                'codeEntryType': 'git',
                'codeEntryAttributes': {
                    'workDir': 'path/inside/repo',
                    'reference': 'refs/heads/my-branch',
                    'username': '',
                    'password': 'my-access-token'
                }
            }
        }
    }


def test_load_function_from_remote_source_s3():
    fn = mlrun.new_function("test-func", kind="nuclio")
    fn.from_remote_source("s3://my-bucket/path/in/bucket/my-functions-archive",
                          handler="path/inside/functions/archive#main:Handler",
                          runtime="golang",
                          credentials={"AWS_ACCESS_KEY_ID": "some-id", "AWS_SECRET_ACCESS_KEY": "some-secret"})

    assert fn.spec.base_spec == {
        'apiVersion': 'nuclio.io/v1',
        'kind': 'Function',
        'metadata': {'name': 'notebook', 'labels': {}, 'annotations': {}},
        'spec': {
            'runtime': 'golang',
            'handler': 'main:Handler',
            'env': [],
            'volumes': [],
            'build': {
                'commands': [],
                'noBaseImagesPull': True,
                'path': 's3://my-bucket/path/in/bucket/my-functions-archive',
                'codeEntryType': 's3',
                'codeEntryAttributes': {
                    'workDir': 'path/inside/functions/archive',
                    's3Bucket': 'my-bucket',
                    's3ItemKey': 'path/in/bucket/my-functions-archive',
                    's3AccessKeyId': 'some-id',
                    's3SecretAccessKey': 'some-secret',
                    's3SessionToken': ''
                }
            }
        }
    }


def test_load_function_from_remote_source_v3io():
    fn = mlrun.new_function("test-func", kind="nuclio")
    fn.from_remote_source("v3ios://host.com/container/my-functions-archive.zip",
                          handler="path/inside/functions/archive#main:handler",
                          credentials={"V3IO_ACCESS_KEY": "ma-access-key"})

    assert fn.spec.base_spec == {
        'apiVersion': 'nuclio.io/v1',
        'kind': 'Function',
        'metadata': {'name': 'notebook', 'labels': {}, 'annotations': {}},
        'spec': {
            'runtime': 'python:3.7',
            'handler': 'main:handler',
            'env': [],
            'volumes': [],
            'build': {
                'commands': [],
                'noBaseImagesPull': True,
                'path': 'https://host.com/container/my-functions-archive.zip',
                'codeEntryType': 'archive',
                'codeEntryAttributes': {
                    'workDir': 'path/inside/functions/archive',
                    'headers': {'headers': {'X-V3io-Session-Key': 'ma-access-key'}}
                }
            }
        }
    }


def test_generate_nuclio_volumes():
    volume_1_name = "volume-name"
    volume_1 = {
        "name": volume_1_name,
        "flexVolume": {
            "driver": "v3io/fuse",
            "options": {
                "container": "users",
                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                "subPath": "/pipelines/.mlrun",
            },
        },
    }
    volume_2_name = "second-volume-name"
    volume_2 = {
        "name": volume_2_name,
        "secret": {"secretName": "secret-name"},
    }
    volume_1_volume_mount_1 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/path",
    }
    volume_1_volume_mount_2 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/2/path",
    }
    volume_2_volume_mount_1 = {
        "name": volume_2_name,
        "mountPath": "/secret/second/volume/mount/path",
    }
    runtime = {
        "kind": "nuclio",
        "metadata": {"name": "some-function", "project": "default"},
        "spec": {
            "volumes": [volume_1, volume_2],
            "volume_mounts": [
                volume_1_volume_mount_1,
                volume_1_volume_mount_2,
                volume_2_volume_mount_1,
            ],
        },
    }
    expected_nuclio_volumes = [
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_1},
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_2},
        {"volume": volume_2, "volumeMount": volume_2_volume_mount_1},
    ]
    function = mlrun.new_function(runtime=runtime)
    nuclio_volumes = function.spec.generate_nuclio_volumes()
    assert DeepDiff(expected_nuclio_volumes, nuclio_volumes, ignore_order=True,) == {}


if __name__ == '__main__':
    test_load_function_from_remote_source_git()
    test_load_function_from_remote_source_s3()
    test_load_function_from_remote_source_v3io()
