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
@click.option('--params', '-p', default='', help='parameters')
@click.option('--artifact', '-a', multiple=True, help='input artifact')
@click.option('--secrets', '-s', default='', help='secrets file')
#@click.option('--secrets', '-s', type=click.File(), help='secrets file')
def run(file, params, artifact, secrets):
    """Execute a task and inject parameters."""

    spec_dict = {}
    if params:
        spec_dict['parameters'] = literal_eval(params)
    if artifact:
        spec_dict['input_artifacts'] = line2keylist(artifact)
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
        out += {keyname: key, valname: value}
    return out


if __name__ == "__main__":
    main()


