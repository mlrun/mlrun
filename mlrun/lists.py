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


import pandas as pd

from .config import config
from .utils import get_in, flatten
from .render import runs_to_html, artifacts_to_html


class RunList(list):

    def to_rows(self):
        rows = []
        head = ['project', 'uid', 'iter', 'start', 'state', 'name', 'labels',
                'inputs', 'parameters', 'results', 'artifacts', 'error']
        for run in self:
            row = [
                get_in(run, 'metadata.project', config.default_project),
                get_in(run, 'metadata.uid', ''),
                get_in(run, 'metadata.iteration', ''),
                get_in(run, 'status.start_time', ''),
                get_in(run, 'status.state', ''),
                get_in(run, 'metadata.name', ''),
                get_in(run, 'metadata.labels', ''),
                get_in(run, 'spec.inputs', ''),
                get_in(run, 'spec.parameters', ''),
                get_in(run, 'status.results', ''),
                get_in(run, 'status.artifacts', []),
                get_in(run, 'status.error', ''),
            ]
            rows.append(row)

        return [head] + rows

    def to_df(self, flat=False):
        rows = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0])  # .set_index('iter')
        df['start'] = pd.to_datetime(df['start'])

        if flat:
            df = flatten(df, 'labels')
            df = flatten(df, 'parameters', 'param_')
            df = flatten(df, 'results', 'out_')

        return df

    def show(self, display=True, classes=None, short=False):
        html = runs_to_html(self.to_df(), display, classes=classes, short=short)
        if not display:
            return html


class ArtifactList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.tag = ''

    def to_rows(self):
        rows = []
        head = {'tree': '', 'key': '', 'iter': '', 'kind': '', 'path': 'target_path', 'hash': '',
                'viewer': '', 'updated': '', 'description': '', 'producer': '',
                'sources': '', 'labels': ''}
        for artifact in self:
            row = [get_in(artifact, v or k, '') for k, v in head.items()]
            rows.append(row)

        return [head.keys()] + rows

    def to_df(self, flat=False):
        rows = self.to_rows()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df['updated'] = pd.to_datetime(df['updated'])

        if flat:
            df = flatten(df, 'producer', 'prod_')
            df = flatten(df, 'sources', 'src_')

        return df

    def show(self, display=True, classes=None):
        df = self.to_df()
        if self.tag != '*':
            df.drop('tree', axis=1, inplace=True)
        html = artifacts_to_html(df, display, classes=classes)
        if not display:
            return html


class FunctionList(list):
    def __init__(self):
        pass
        # TODO
