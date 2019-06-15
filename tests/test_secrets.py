from os import environ

from mlrun.secrets import SecretsStore

spec = {
    'secret_sources': [
        {'kind': 'file', 'source': 'tests/secrets_test.txt'},
        {'kind': 'inline', 'source': {'abc':'def'}},
        {'kind': 'env', 'source': 'ENV123,ENV456'},
    ],
}


def test_load():
    environ['ENV123'] = 'xx'
    environ['ENV456'] = 'yy'
    ss = SecretsStore()
    ss.from_dict(spec)

    assert ss.get('ENV123') == 'xx', 'failed on 1st env var secret'
    assert ss.get('ENV456') == 'yy', 'failed on 1st env var secret'
    assert ss.get('MYENV') == '123', 'failed on 1st env var secret'
    assert ss.get('MY2NDENV') == '456', 'failed on 1st env var secret'
    assert ss.get('abc') == 'def', 'failed on 1st env var secret'
    print(ss.get_all())

def test_inline_str():
    ss = SecretsStore()
    spec = {
        'secret_sources': [
            {'kind': 'inline', 'source': "{'abc': 'def'}"},
        ],
    }

    ss.from_dict(spec)
    assert ss.get('abc') == 'def', 'failed on 1st env var secret'



