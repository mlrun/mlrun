import mlrun
import pytest
import yaml
import os
import random
from pathlib import Path

here = Path(__file__).absolute().parent
config_file_path = here / 'test-azure-blob.yml'
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)


def azure_connection_configured():
    return config['env'].get('AZURE_STORAGE_CONNECTION_STRING') is not None


def prepare_env():
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = config['env'].get(
        'AZURE_STORAGE_CONNECTION_STRING'
    )


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason='This is an integration test, add the needed environment variables in test-azure-blob.yml '
    'to run it',
)
def test_azure_blob():
    prepare_env()

    blob_path = 'az://' + config['env'].get('AZURE_CONTAINER')
    blob_dir = 'test_mlrun_azure_blob'
    test_filename = here / 'test.txt'
    with open(test_filename, 'r') as f:
        test_string = f.read()

    blob_file = 'file_{0}.txt'.format(random.randint(0, 1000))
    blob_url = blob_path + '/' + blob_dir + '/' + blob_file
    print(f'\nBlob URL: {blob_url}')

    data_item = mlrun.run.get_dataitem(blob_url)
    data_item.put(test_string)

    response = data_item.get()
    assert response.decode() == test_string, 'Result differs from original test'

    response = data_item.get(offset=20)
    assert response.decode() == test_string[20:], 'Partial result not as expected'

    stat = data_item.stat()
    assert stat.size == len(test_string), 'Stat size different than expected'

    # Check dir list
    dir_data_item = mlrun.run.get_dataitem(blob_path + '/' + blob_dir)
    dir_list = dir_data_item.listdir()
    assert any(
        blob_file in item for item in dir_list
    ), 'List dir did not contain our file name'


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason='This is an integration test, add the needed environment variables in test-azure-blob.yml '
    'to run it',
)
def test_blob_upload():
    # Check upload functionality
    prepare_env()

    blob_path = 'az://' + config['env'].get('AZURE_CONTAINER')
    blob_dir = 'test_mlrun_azure_blob'
    test_filename = here / 'test.txt'
    blob_file = 'file_{0}.txt'.format(random.randint(0, 1000))
    blob_url = blob_path + '/' + blob_dir + '/' + blob_file
    print(f'\nBlob URL: {blob_url}')

    upload_data_item = mlrun.run.get_dataitem(blob_url)
    upload_data_item.upload(test_filename)

    response = upload_data_item.get()

    with open(test_filename, 'r') as f:
        test_string = f.read()

    assert response.decode() == test_string, 'Result differs from original test'
