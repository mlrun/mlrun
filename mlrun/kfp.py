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

    outputs = []
    for output in struct['status'][run_keys.output_artifacts]:
        key = output["key"]
        target = output.get('target_path', '')
        try:
            with open(f'/tmp/{key}', 'w') as fp:
                fp.write(target)
        except:
            pass

        if target.startswith('v3io:///'):
            target = target.replace('v3io:///', 'http://v3io-webapi:8081/')

        viewer = output.get('viewer', '')
        if viewer in ['web-app', 'chart']:
            meta = {'type': 'web-app',
                    'source': target}
            outputs += [meta]

        elif viewer == 'table':
            header = output.get('header', None)
            if header and target.endswith('.csv'):
                meta = {'type': 'table',
                    'format': 'csv',
                    'header': header,
                    'source': target}
                outputs += [meta]

    text = yaml.dump(struct, default_flow_style=False, sort_keys=False)
    text = "# Run Report\n```yaml\n" + text + "```\n"
    metadata = {
        'outputs': outputs + [{
            'type': 'markdown',
            'storage': 'inline',
            'source': text
        }]
    }
    with open(KFPMETA_DIR + 'mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


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