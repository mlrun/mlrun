import pytest
import yaml

from conftest import here
from mlrun import funcdoc


def test_func_info_ann():
    def inc(n: int) -> int:
        """increment n"""
        return n + 1

    out = funcdoc.func_info(inc)
    expected = {
        'name': inc.__name__,
        'doc': inc.__doc__,
        'return': {'type': 'int', 'doc': ''},
        'params': [
            {'name': 'n', 'type': 'int', 'doc': ''},
        ],
    }
    assert out == expected, 'inc'


def test_func_info_no_ann():
    def inc(n):
        return n + 1

    out = funcdoc.func_info(inc)
    expected = {
        'name': inc.__name__,
        'doc': '',
        'return': {'type': '', 'doc': ''},
        'params': [
            {'name': 'n', 'type': '', 'doc': ''},
        ],
    }
    assert out == expected, 'inc'


def load_cases(name):
    with open(here / name) as fp:
        data = yaml.load(fp)

    for i, case in enumerate(data):
        name = case.get('name', '')
        tid = f'{i} - {name}'
        yield pytest.param(case['text'], case['expected'], id=tid)


@pytest.mark.parametrize('text, expected', load_cases('rst_cases.yml'))
def test_rst(text, expected):
    doc, params, ret = funcdoc.parse_rst(text)
    assert expected['doc'].strip() == doc.strip(), 'doc'
    assert expected['params'] == params, 'params'
    assert expected['ret'] == ret, 'ret'


# gdoc = '''
# Summary line.

# Extended description of function.

# Args:
#     arg1: Description of arg1
#     arg2: Description of arg2

# Returns:
#     Description of return value
# '''
