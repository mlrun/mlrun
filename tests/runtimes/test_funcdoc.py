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
from textwrap import dedent

import pytest
import yaml

from mlrun.runtimes import funcdoc
from tests.conftest import tests_root_directory


def load_rst_cases(name):
    with open(tests_root_directory / "runtimes" / name) as fp:
        data = yaml.load(fp)

    for i, case in enumerate(data):
        name = case.get("name", "")
        tid = f"{i} - {name}"
        yield pytest.param(case["text"], case["expected"], id=tid)


@pytest.mark.parametrize("text, expected", load_rst_cases("rst_cases.yml"))
def test_rst(text, expected):
    doc, params, ret = funcdoc.parse_rst(text)
    assert expected["doc"].strip() == doc.strip(), "doc"
    assert expected["params"] == params, "params"
    assert expected["ret"] == ret, "ret"


def is_ast_func(obj):
    return isinstance(obj, ast.FunctionDef)


def ast_func(code):
    funcs = [s for s in ast.parse(code).body if is_ast_func(s)]
    assert len(funcs) == 1, f"{len(funcs)} functions in:\n{code}"
    return funcs[0]


def eval_func(code):
    out = {}
    exec(code, None, out)
    funcs = [obj for obj in out.values() if callable(obj)]
    assert len(funcs) == 1, f"more than one function in:\n{code}"
    return funcs[0]


info_handlers = [
    (funcdoc.func_info, eval_func),
    (funcdoc.ast_func_info, ast_func),
]


def load_info_cases():
    with open(tests_root_directory / "runtimes" / "info_cases.yml") as fp:
        cases = yaml.load(fp)

    for case in cases:
        for info_fn, conv in info_handlers:
            obj = conv(case["code"])
            tid = f'{case["id"]}-{info_fn.__name__}'
            expected = case["expected"].copy()
            # No line info in evaled functions
            if info_fn is funcdoc.func_info:
                expected["lineno"] = -1
            yield pytest.param(info_fn, obj, expected, id=tid)


@pytest.mark.parametrize("info_fn, obj, expected", load_info_cases())
def test_func_info(info_fn, obj, expected):
    out = info_fn(obj)
    assert expected == out


find_handlers_code = """
def dec(n):
    return n - 1

# mlrun:handler
def inc(n):
    return n + 1
"""

find_handlers_expected = [
    {
        "name": "inc",
        "doc": "",
        "return": funcdoc.param_dict(),
        "params": [funcdoc.param_dict("n")],
        "lineno": 6,
    },
]


def test_find_handlers():
    funcs = funcdoc.find_handlers(find_handlers_code)
    assert find_handlers_expected == funcs


ast_code_cases = [
    "{'x': 1, 'y': 2}",
    "dict(x=1, y=2)",
    "{}",
    "[1, 2]",
    "[]",
    "(1, 2)",
    "()",
    "{1, 2}",
    "set()",
    "Point(1, 2)",
    "3",
    "'hello'",
    "None",
]


@pytest.mark.parametrize("expr", ast_code_cases)
def test_ast_code(expr):
    node = ast.parse(expr).body[0].value
    code = funcdoc.ast_code(node)
    assert expr == code


def test_ast_none():
    code = """
    def fn() -> None:
        pass
    """
    fn = ast.parse(dedent(code)).body[0]
    funcdoc.ast_func_info(fn)


def test_ast_compound():
    with open(f"{tests_root_directory}/runtimes/arc.txt") as fp:
        code = fp.read()

    fn = ast_func(code)
    info = funcdoc.ast_func_info(fn)
    for i, param in enumerate(info["params"]):
        if i in (4, 8):
            continue
        assert param["type"], f"{i}: {param}"


underscore_code = """
def info(message):
    _log('INFO', message)

def warning(message):
    _log('WARNING', message)

def _log(level, message):
    print(f'{level} - {message}')
"""


def test_ignore_underscore():
    funcs = funcdoc.find_handlers(underscore_code)
    names = {fn["name"] for fn in funcs}
    assert {"info", "warning"} == names, "names"


def test_annotate_mod():
    code = """
    import mlrun

    def handler(data: mlrun.DataItem):
        ...
    """

    handlers = funcdoc.find_handlers(dedent(code))
    param = handlers[0]["params"][0]
    assert param["type"] == "DataItem"
