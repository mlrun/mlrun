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
import inspect
import re


def type_name(ann):
    if ann is inspect.Signature.empty:
        return ''
    return getattr(ann, '__name__', str(ann))


def inspect_param(param: inspect.Parameter) -> dict:
    return param_dict(param.name, type_name(param.annotation))


def param_dict(name='', type='', doc=''):
    return {
        'name': name,
        'type': type,
        'doc': doc,
    }


def func_dict(name, doc, params, returns, lineno):
    return {
        'name': name,
        'doc': doc,
        'params': params,
        'return': returns,
        'lineno': lineno,
    }


def func_info(fn) -> dict:
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ''

    out = func_dict(
        name=fn.__name__,
        doc=doc,
        params=[inspect_param(p) for p in sig.parameters.values()],
        returns=param_dict(type=type_name(sig.return_annotation), doc=doc),
        lineno=func_lineno(fn),
    )

    if not fn.__doc__ or not fn.__doc__.strip():
        return out

    return merge_doc(out, doc)


def func_lineno(fn):
    try:
        return inspect.getsourcelines(fn)[1]
    except (TypeError, OSError):
        return -1


def merge_doc(out, doc):
    doc, params, ret = parse_rst(doc)
    out['doc'] = doc

    # TODO: Check that doc matches signature
    for tparam, param in zip(out['params'], params):
        tparam['doc'] = param['doc']
        if not tparam['type']:
            tparam['type'] = param['type']
    out['return']['doc'] = ret['doc']
    if not out['return']['type']:
        out['return']['type'] = ret['type']
    return out


def rst_read_doc(lines):
    doc = []
    for i, line in enumerate(lines):
        if line[:1] == ':':
            return i, '\n'.join(doc).strip()
        doc.append(line)
    return -1, '\n'.join(doc).strip()


def rst_read_section(lines, i):
    # Skip empty lines
    while not lines[i].strip() and i < len(lines):
        i += 1

    # :param path: The path of the file to wrap
    match = re.match(r':(\w+)(\s+\w+)?:', lines[i])
    if not match:
        raise ValueError(f'{i}: bad line - {lines[i]!r}')

    tag = match.group(1)
    value = match.group(2).strip() if match.group(2) else ''
    text = lines[i][match.end():].lstrip()
    for i in range(i+1, len(lines)):
        if re.match(r'\t+| {3,}', lines[i]):
            text += ' ' + lines[i].lstrip()
        else:
            return tag, value, text, i
    return tag, value, text.strip(), -1


def parse_rst(docstring: str):
    lines = docstring.splitlines()
    i, doc = rst_read_doc(lines)
    params, names = {}, []
    ret = param_dict()

    while i != -1:
        tag, value, text, i = rst_read_section(lines, i)
        if tag == 'param':
            params[value] = param_dict(name=value, doc=text)
            names.append(value)
        elif tag == 'type':
            # TODO: Check param
            params[value]['type'] = text
        elif tag == 'returns':
            ret['doc'] = text
        elif tag == 'rtype':
            ret['type'] = text
        else:
            raise ValueError(f'{i+1}: uknown tag - {lines[i]!r}')

    params = [params[name] for name in names]
    return doc, params, ret


def ast_func_info(func: ast.FunctionDef):
    doc = ast_doc(func)
    rtype = func.returns.id if func.returns else ''

    out = func_dict(
        name=func.name,
        doc=doc,
        params=[ast_param_dict(p) for p in func.args.args],
        returns=param_dict(type=rtype),
        lineno=func.lineno,
    )

    if not doc.strip():
        return out

    return merge_doc(out, doc)


def ast_doc(func: ast.FunctionDef) -> str:
    if not func.body:
        return ''

    if not isinstance(func.body[0], ast.Expr):
        return ''

    if not isinstance(func.body[0].value, ast.Str):
        return ''

    return inspect.cleandoc(func.body[0].value.s)


def ast_param_dict(param: ast.arg) -> dict:
    return {
        'name': param.arg,
        'type': param.annotation.id if param.annotation else '',
        'doc': '',
    }


class FunctionFinder(ast.NodeVisitor):
    def __init__(self):
        self.funcs = []

    def visit_FunctionDef(self, node):
        self.funcs.append(node)
        self.generic_visit(node)


def find_functions(code: str):
    mod = ast.parse(code)
    finder = FunctionFinder()
    finder.visit(mod)
    funcs = [ast_func_info(fn) for fn in finder.funcs]
    markers = find_handler_markers(code)
    return filter_funcs(funcs, markers)


def filter_funcs(funcs, markers):
    markers = list(markers)
    if not markers:
        return funcs

    return [func for func in funcs if is_marked(func, markers)]


def is_marked(func, markers):
    for marker in markers:
        if func['lineno'] - marker == 1:
            return True
    return False


def find_handler_markers(code: str):
    for lnum, line in enumerate(code.splitlines(), 1):
        # '# mlrun:handler'
        if re.match(r'#\s*mlrun:handler', line):
            yield lnum
