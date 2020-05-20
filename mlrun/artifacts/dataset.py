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
import os
import pathlib
from io import StringIO
from tempfile import mktemp

from pandas.io.json import build_table_schema

from .base import Artifact

preview_lines = 20
max_csv = 10000


class TableArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['schema', 'header']
    kind = 'table'

    def __init__(self, key=None, body=None, df=None, viewer=None, visible=False,
                 inline=False, format=None, header=None, schema=None):

        if key:
            key_suffix = pathlib.Path(key).suffix
            if not format and key_suffix:
                format = key_suffix[1:]
        super().__init__(
            key, body, viewer=viewer, is_inline=inline, format=format)

        if df is not None:
            self._is_df = True
            self.header = df.columns.values.tolist()
            self.format = 'csv' # todo other formats
            # if visible and not key_suffix:
            #     key += '.csv'
            self._body = df
        else:
            self._is_df = False
            self.header = header

        self.schema = schema
        if not viewer:
            viewer = 'table' if visible else None
        self.viewer = viewer

    def get_body(self):
        if not self._is_df:
            return self._body
        csv_buffer = StringIO()
        self._body.to_csv(
            csv_buffer, index=False, line_terminator='\n', encoding='utf-8')
        return csv_buffer.getvalue()


supported_formats = ['csv', 'parquet', 'pq', 'tsdb', 'kv']


class DatasetArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + [
        'schema', 'header', 'length', 'preview', 'stats', 'analysis']
    kind = 'dataset'

    def __init__(self, key=None, df=None, preview=None, format='',
                 stats=None, target_path=None, analysis=None, **kwargs):

        format = format.lower()
        super().__init__(key, None, format=format, target_path=target_path)
        if format and format not in supported_formats:
            raise ValueError('unsupported format {} use one of {}'.format(
                format, '|'.join(supported_formats)))

        if format == 'pq':
            format = 'parquet'
        self.format = format
        self.stats = None
        self.analysis = analysis or {}

        if df is not None:
            self.length = df.shape[0]
            preview = preview or preview_lines
            shortdf = df
            if self.length > preview:
                shortdf = df.head(preview)
            self.header = shortdf.reset_index().columns.values.tolist()
            self.preview = shortdf.reset_index().values.tolist()
            self.schema = build_table_schema(df)
            if stats or self.length < max_csv:
                self.stats = get_stats(df)

        self._df = df
        self._kw = kwargs

    def get_body(self):
        csv_buffer = StringIO()
        self._body.to_csv(
            csv_buffer, index=False, line_terminator='\n', encoding='utf-8')
        return csv_buffer.getvalue()

    def upload(self, data_stores):
        suffix = pathlib.Path(self.target_path).suffix
        if not self.format:
            if suffix and suffix in ['.csv', '.parquet', '.pq']:
                self.format = 'csv' if suffix == '.csv' else 'parquet'
            else:
                self.format = 'csv' if self.length < max_csv else 'parquet'

        src_path = self.src_path
        if src_path and os.path.isfile(src_path):
            self._upload_file(src_path, data_stores)
            return

        if self._df is None:
            return

        if self.format in ['csv', 'parquet']:
            if not suffix:
                self.target_path = self.target_path + '.' + self.format
            writer_string = 'to_{}'.format(self.format)
            saving_func = getattr(self._df, writer_string, None)
            target = self.target_path
            to_upload = False
            if '://' in target:
                target = mktemp()
                to_upload = True
            else:
                dir = os.path.dirname(target)
                if dir:
                    os.makedirs(dir, exist_ok=True)

            saving_func(target, **self._kw)
            if to_upload:
                self._upload_file(target, data_stores)
                self._set_meta(target)
                os.remove(target)
            else:
                self._set_meta(target)
            return

        raise ValueError(f'format {self.format} not implemented yes')


def get_stats(df):
    d = {}
    for k, v in df.describe(include='all').items():
        v = {m: float(x) if not isinstance(x, str) else x
             for m, x in v.dropna().items()}
        d[k] = v
    return d
