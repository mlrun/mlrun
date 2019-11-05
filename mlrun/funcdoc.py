import inspect
import re


def type_name(ann):
    if ann is inspect.Signature.empty:
        return ''
    return getattr(ann, '__name__', str(ann))


def param_dict(param: inspect.Parameter) -> dict:
    return {
        'name': param.name,
        'type': type_name(param.annotation),
        'doc': '',
    }


def func_info(fn) -> dict:
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ''

    out = {
        'name': fn.__name__,
        'doc': doc,
        'params': [param_dict(p) for p in sig.parameters.values()],
        'return': {
            'type': type_name(sig.return_annotation),
            'doc': '',
        },
    }

    if not fn.__doc__ or not fn.__doc__.strip():
        return out

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
    ret = {'doc': '', 'type': ''}

    while i != -1:
        tag, value, text, i = rst_read_section(lines, i)
        if tag == 'param':
            params[value] = {'name': value, 'doc': text}
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
