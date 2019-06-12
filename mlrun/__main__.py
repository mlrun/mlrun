from os import path

import click
import json
import os
from ast import literal_eval
import getpass



@click.group()
def main():
    pass

@main.command()
@click.argument("file", type=click.File())
@click.option('--param', '-p', default='', multiple=True,
              help="parameter name and value tuples, e.g. -p x=37 -p y='text'")
@click.option('--in-artifact', '-i', multiple=True, help='input artifact')
@click.option('--out-artifact', '-o', multiple=True, help='output artifact')
@click.option('--secrets', '-s', default='', help='secrets file')
#@click.option('--secrets', '-s', type=click.File(), help='secrets file')
def run(file, param, in_artifact, out_artifact, secrets):
    """Execute a task and inject parameters."""

    spec_dict = {}
    if param:
        params_dict = {}
        for param in param:
            i = param.find('=')
            if i == -1:
                continue
            key, value = param[:i].strip(), param[i + 1:].strip()
            if key is None:
                raise ValueError(f'cannot find param key in line ({param})')
            params_dict[key] = literal_eval(value)
        spec_dict['parameters'] = params_dict
    if in_artifact:
        spec_dict['input_artifacts'] = line2keylist(in_artifact)
    if out_artifact:
        spec_dict['output_artifacts'] = line2keylist(out_artifact)
    if secrets:
        spec_dict['secret_sources'] = [{'kind':'file', 'source': secrets}]

    if spec_dict:
        meta = {'parent_type': 'local', 'owner': getpass.getuser()}
        os.environ['MLRUN_EXEC_CONFIG'] = json.dumps({'spec': spec_dict})

    exec(file.read())


def line2keylist(lines: list, keyname='key', valname='path'):
    out = []
    for line in lines:
        i = line.find('=')
        if i == -1:
            continue
        key, value = line[:i].strip(), line[i + 1:].strip()
        if key is None:
            raise ValueError('cannot find key in line (key=value)')
        value = path.expandvars(value)
        out += [{keyname: key, valname: value}]
    return out


if __name__ == "__main__":
    main()


