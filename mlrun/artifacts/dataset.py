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

import pathlib
from io import StringIO

from .base import Artifact


class TableArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['schema', 'header']
    kind = 'table'

    def __init__(self, key, body=None, df=None, viewer=None, visible=False,
                 inline=False, format=None, header=None, schema=None):

        key_suffix = pathlib.Path(key).suffix
        if not format and key_suffix:
            format = key_suffix[1:]
        super().__init__(
            key, body, viewer=viewer, inline=inline, format=format)

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


