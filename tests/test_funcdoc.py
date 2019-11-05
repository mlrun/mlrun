# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def test_func_info_ann_doc():
    def inc(n: int) -> int:
        """increment n

        :param n: number to increment
        :returns: a number
        :rtype: int
        """
        return n + 1

    out = funcdoc.func_info(inc)
    expected = {
        'name': inc.__name__,
        'doc': 'increment n',
        'return': {'type': 'int', 'doc': 'a number'},
        'params': [
            {'name': 'n', 'type': 'int', 'doc': 'number to increment'},
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
