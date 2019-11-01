
import pandas as pd
from .utils import get_in, flatten
from .render import runs_to_html, artifacts_to_html


class RunList(list):

    def to_rows(self):
        rows = []
        head = ['uid', 'iter', 'start', 'state', 'name', 'labels',
                'inputs', 'parameters', 'results', 'artifacts', 'error']
        for run in self:
            row = [
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
        df = pd.DataFrame(rows[1:], columns=rows[0]) #.set_index('iter')
        df['start'] = pd.to_datetime(df['start'])

        if flat:
            df = flatten(df, 'labels')
            df = flatten(df, 'parameters', 'param_')
            df = flatten(df, 'results', 'out_')

        return df

    def show(self, display=True, classes=None):
        html = runs_to_html(self.to_df(), display, classes=classes)
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
        df['updated'] = pd.to_datetime(df['updated'], unit='s')

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
