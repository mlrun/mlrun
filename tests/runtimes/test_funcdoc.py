# Copyright 2023 Iguazio
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
        data = yaml.safe_load(fp)

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


def load_info_cases():
    with open(tests_root_directory / "runtimes" / "info_cases.yml") as fp:
        cases = yaml.safe_load(fp)

    for case in cases:
        obj = ast_func(case["code"])
        expected = case["expected"].copy()
        yield pytest.param(funcdoc.ast_func_info, obj, expected, id=case["id"])


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
        "return": funcdoc.param_dict(type=None, default=None),
        "params": [funcdoc.param_dict("n", default=None)],
        "lineno": 6,
        "has_varargs": False,
        "has_kwargs": False,
    },
]


def test_find_handlers():
    funcs = funcdoc.find_handlers(find_handlers_code)
    assert funcs == find_handlers_expected


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
    fn: ast.FunctionDef = ast.parse(dedent(code)).body[0]
    funcdoc.ast_func_info(fn)


@pytest.mark.parametrize(
    "func_code,expected_has_varargs,expected_has_kwargs",
    [
        (
            """
    def fn(p1,p2,*args,**kwargs) -> None:
        pass
    """,
            True,
            True,
        ),
        (
            """
    def fn(p1,p2,*args) -> None:
        pass
    """,
            True,
            False,
        ),
        (
            """
    def fn(p1,p2,**kwargs) -> None:
        pass
    """,
            False,
            True,
        ),
        (
            """
    def fn(p1,p2) -> None:
        pass
    """,
            False,
            False,
        ),
        (
            """
    def fn(p1,p2,**something) -> None:
        pass
    """,
            False,
            True,
        ),
    ],
)
def test_ast_func_info_with_kwargs_and_args(
    func_code, expected_has_varargs, expected_has_kwargs
):
    fn: ast.FunctionDef = ast.parse(dedent(func_code)).body[0]
    func_info = funcdoc.ast_func_info(fn)
    assert func_info["has_varargs"] == expected_has_varargs
    assert func_info["has_kwargs"] == expected_has_kwargs

    func = funcdoc.find_handlers(dedent(func_code))[0]
    assert func["has_varargs"] == expected_has_varargs
    assert func["has_kwargs"] == expected_has_kwargs


def test_ast_compound():
    param_types = []
    with open(f"{tests_root_directory}/runtimes/arc.txt") as fp:
        code = fp.read()

        # collect the types of the function parameters
        # assumes each param is in a new line for simplicity
        for line in code.splitlines()[3:15]:
            if ":" not in line:
                param_types.append(None)
                continue

            param_type = line[line.index(":") + 1 :]
            if "=" in param_type:
                param_type = param_type[: param_type.index("=")]
            param_type = param_type[:-1].strip()
            param_types.append(param_type)

    fn = ast_func(code)
    info = funcdoc.ast_func_info(fn)
    for i, param in enumerate(info["params"]):
        if i in (4, 8):
            continue
        assert (
            param["type"] == param_types[i]
        ), f"param at index {i} has a bad type value. param: {param}"


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


@pytest.mark.parametrize(
    "func_code,expected_return_type",
    [
        pytest.param(
            """
    def fn1():
        pass
    """,
            None,
            id="no-hint",
        ),
        pytest.param(
            """
    def fn2() -> str:
        pass
    """,
            "str",
            id="str",
        ),
        pytest.param(
            """
    def fn3() -> Union[str, pd.DataFrame]:
        pass
    """,
            "Union[str, pd.DataFrame]",
            id="union",
        ),
        pytest.param(
            """
    def fn4() -> None:
        pass
    """,
            "None",
            id="None",
        ),
    ],
)
def test_return_types(func_code, expected_return_type):
    fn: ast.FunctionDef = ast.parse(dedent(func_code)).body[0]
    func_info = funcdoc.ast_func_info(fn)
    assert func_info["return"]["type"] == expected_return_type
