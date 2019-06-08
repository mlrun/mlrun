from os import path

import click
import json
import os
from ast import literal_eval
from .utils import list2dict


@click.group()
def main():
    pass

@main.command()
@click.argument("file", type=click.File())
@click.option('--params', '-p', default='', help='parameters')
@click.option('--artifact', '-a', multiple=True, help='input artifact')
@click.option('--secrets', '-s', type=click.File(), help='secrets file')
def run(file, params, artifact, secrets):
    """Print QR code to the terminal."""

    spec_dict = {}
    if params:
        spec_dict['parameters'] = literal_eval(params)
    if artifact:
        spec_dict['input_artifacts'] = list2dict(artifact)

    if spec_dict:
        os.environ['MLRUN_EXEC_CONFIG'] = json.dumps({'spec': spec_dict})

    if secrets:
        lines = secrets.read().splitlines()
        secrets_dict = list2dict(lines)
        os.environ['MLRUN_EXEC_SECRETS'] = json.dumps(secrets_dict)




    exec(file.read())


if __name__ == "__main__":
    main()


