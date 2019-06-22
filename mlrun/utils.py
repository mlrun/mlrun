from os import path

class run_keys:
    input_path = 'default_input_path'
    output_path = 'default_output_path'
    input_objects = 'input_objects'
    output_artifacts = 'output_artifacts'
    data_stores = 'data_stores'
    secrets = 'secret_sources'


def list2dict(lines: list):
    out = {}
    for line in lines:
        i = line.find('=')
        if i == -1:
            continue
        key, value = line[:i].strip(), line[i + 1:].strip()
        if key is None:
            raise ValueError('cannot find key in line (key=value)')
        value = path.expandvars(value)
        out[key] = value
    return out


def uxjoin(base, path):
    if base:
        return '{}/{}'.format(base, path)
    return path
