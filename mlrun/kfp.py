import json
import yaml
from .utils import run_keys

KFPMETA_DIR = '/'

def write_kfpmeta(struct):
    outputs = struct['status']['outputs']
    metrics = {'metrics':
                   [{'name': k, 'numberValue':v } for k, v in outputs.items() if isinstance(v, (int, float, complex))]}
    with open(KFPMETA_DIR + 'mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    text = yaml.dump(struct, default_flow_style=False, sort_keys=False)
    text = "# Run Report\n```yaml\n" + text + "```\n"

    metadata = {
        'outputs': [{
            'type': 'markdown',
            'storage': 'inline',
            'source': text
        }]
    }
    with open(KFPMETA_DIR + 'mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    for output in struct['status'][run_keys.output_artifacts]:
        try:
            key = output["key"]
            with open(f'/tmp/{key}', 'w') as fp:
                fp.write(output["target_path"])
        except:
            pass


def mlrun_op(name='', image='v3io/mlrun', command='', params={}, inputs={}, outputs={}, out_path='', rundb=''):
    from kfp import dsl
    cmd = ['python', '-m', 'mlrun', 'run', '--kfp', '--workflow', '{{workflow.uid}}']
    for p, val in params.items():
        cmd += ['-p', f'{p}={val}']
    for i, val in inputs.items():
        cmd += ['-i', f'{i}={val}']
    file_outputs = {}
    for o, val in outputs.items():
        cmd += ['-o', f'{o}={val}']
        file_outputs[o.replace('.', '-')] = f'/tmp/{o}'
    if out_path:
        cmd += ['--out-path', out_path]
    if rundb:
        cmd += ['--rundb', rundb]

    cop = dsl.ContainerOp(
        name=name,
        image=image,
        command=cmd + [command],
        file_outputs=file_outputs,
    )
    #cop.apply(mount_v3io(container='users', sub_path='/iguazio', mount_path='/User'))
    #cop.apply(v3io_cred())
    return cop