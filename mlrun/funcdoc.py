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


def inspect_default(value):
    if value is inspect.Signature.empty:
        return ''
    return repr(value)


def inspect_param(param: inspect.Parameter) -> dict:
    name = param.name
    typ = type_name(param.annotation)
    default = inspect_default(param.default)
    return param_dict(name, typ, '', default)


# We're using dict and not classes (here and in func_dict) since this goes
# directly to YAML
def param_dict(name='', type='', doc='', default=''):
    return {
        'default': default,
        'doc': doc,
        'name': name,
        'type': type,
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
        returns=param_dict(type=type_name(sig.return_annotation)),
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

    for param in params:
        for out_param in out['params']:
            if out_param['name'] != param['name']:
                continue
            out_param['doc'] = param['doc'] or out_param['doc']
            out_param['type'] = param['type'] or out_param['type']
            break

    out['return']['doc'] = ret['doc'] or out['return']['doc']
    out['return']['type'] = ret['type'] or out['return']['type']
    return out


def rst_read_doc(lines):
    doc = []
    for i, line in enumerate(lines):
        if line[:1] == ':':
            return i, '\n'.join(doc).strip()
        doc.append(line)
    return -1, '\n'.join(doc).strip()


def rst_read_section(lines, i):
    # Skip empty lines/other lines
    for i, line in enumerate(lines[i:], i):
        if not line.strip():
            continue
        # :param path: The path of the file to wrap
        match = re.match(r':\s*(\w+)(\s+\w+)?\s*:', lines[i])
        if match:
            break
    else:
        return None

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
        out = rst_read_section(lines, i)
        if not out:
            break
        tag, value, text, i = out
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
            raise ValueError(f'{i+1}: unknown tag - {lines[i]!r}')

    params = [params[name] for name in names]
    return doc, params, ret


def ast_func_info(func: ast.FunctionDef):
    doc = ast.get_docstring(func) or ''
    rtype = getattr(func.returns, 'id', '')
    params = [ast_param_dict(p) for p in func.args.args]
    defaults = func.args.defaults
    if defaults:
        for param, default in zip(params[-len(defaults):], defaults):
            param['default'] = ast_code(default)

    out = func_dict(
        name=func.name,
        doc=doc,
        params=params,
        returns=param_dict(type=rtype),
        lineno=func.lineno,
    )

    if not doc.strip():
        return out

    return merge_doc(out, doc)


def ast_param_dict(param: ast.arg) -> dict:
    return {
        'name': param.arg,
        'type': ann_type(param.annotation) if param.annotation else '',
        'doc': '',
        'default': '',
    }


def ann_type(ann):
    if hasattr(ann, 'slice'):
        name = ann.value.id
        inner = ', '.join(ann_type(e) for e in iter_elems(ann.slice))
        return f'{name}[{inner}]'

    return getattr(ann, 'id', '')


def iter_elems(ann):
    if hasattr(ann.value, 'elts'):
        return ann.value.elts
    if not hasattr(ann, 'slice'):
        return [ann.value]
    elif hasattr(ann.slice, 'elts'):
        return ann.slice.elts
    elif hasattr(ann.slice, 'value'):
        return [ann.slice.value]
    return []


class ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.funcs = []
        self.exprs = []

    def generic_visit(self, node):
        self.exprs.append(node)
        super().generic_visit(node)

    def visit_FunctionDef(self, node):
        self.funcs.append(node)
        self.generic_visit(node)


def find_handlers(code: str, handlers=None):
    handlers = set() if handlers is None else set(handlers)
    mod = ast.parse(code)
    visitor = ASTVisitor()
    visitor.visit(mod)
    funcs = [ast_func_info(fn) for fn in visitor.funcs]
    if handlers:
        return [f for f in funcs if f['name'] in handlers]
    else:
        markers = find_handler_markers(code)
        return filter_funcs(funcs, markers)


def filter_funcs(funcs, markers):
    markers = list(markers)
    if not markers:
        return [func for func in funcs if func['name'][0] != '_']

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


def ast_code(expr):
    # Sadly, not such built in
    children = None
    if isinstance(expr, ast.Dict):
        children = zip(expr.keys, expr.values)
        children = [f'{ast_code(k)}: {ast_code(v)}' for k, v in children]
        inner = ', '.join(children)
        return f'{{{inner}}}'
    elif isinstance(expr, ast.Set):
        start, end, children = '{', '}', expr.elts
        if not children:
            return 'set()'
    elif isinstance(expr, ast.Tuple):
        start, end, children = '(', ')', expr.elts
    elif isinstance(expr, ast.List):
        start, end, children = '[', ']', expr.elts
    elif isinstance(expr, ast.Call):
        children = [ast_code(e) for e in expr.args]
        children += [f'{k.arg}={ast_code(k.value)}' for k in expr.keywords]
        inner = ', '.join(children)
        return f'{expr.func.id}({inner})'
    else:  # Leaf (number, str ...)
        return repr(getattr(expr, expr._fields[0]))

    inner = ', '.join(ast_code(e) for e in children)
    return f'{start}{inner}{end}'
