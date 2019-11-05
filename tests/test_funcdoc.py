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

import ast

import pytest
import yaml
from conftest import here
from mlrun import funcdoc


def load_rst_cases(name):
    with open(here / name) as fp:
        data = yaml.load(fp)

    for i, case in enumerate(data):
        name = case.get('name', '')
        tid = f'{i} - {name}'
        yield pytest.param(case['text'], case['expected'], id=tid)


@pytest.mark.parametrize('text, expected', load_rst_cases('rst_cases.yml'))
def test_rst(text, expected):
    doc, params, ret = funcdoc.parse_rst(text)
    assert expected['doc'].strip() == doc.strip(), 'doc'
    assert expected['params'] == params, 'params'
    assert expected['ret'] == ret, 'ret'


def is_ast_func(obj):
    return isinstance(obj, ast.FunctionDef)


def ast_func(code):
    funcs = [s for s in ast.parse(code).body if is_ast_func(s)]
    assert len(funcs) == 1, f'more than one function in:\n{code}'
    return funcs[0]


def eval_func(code):
    out = {}
    exec(code, None, out)
    funcs = [obj for obj in out.values() if callable(obj)]
    assert len(funcs) == 1, f'more than one function in:\n{code}'
    return funcs[0]


info_handlers = [
    (funcdoc.func_info, eval_func),
    (funcdoc.ast_func_info, ast_func),
]


def load_info_cases():
    with open(here / 'info_cases.yml') as fp:
        cases = yaml.load(fp)

    for case in cases:
        for info_fn, conv in info_handlers:
            obj = conv(case['code'])
            tid = f'{case["id"]}-{info_fn.__name__}'
            yield pytest.param(info_fn, obj, case['expected'], id=tid)


@pytest.mark.parametrize('info_fn, obj, expected', load_info_cases())
def test_func_info(info_fn, obj, expected):
    out = info_fn(obj)
    assert out == expected
