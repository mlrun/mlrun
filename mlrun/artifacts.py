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
import os
import hashlib
import base64
from io import BytesIO, StringIO
import pathlib

from .datastore import StoreManager
from .db import RunDBInterface
from .utils import uxjoin, run_keys, dict_to_json
from .model import ModelObj


def file_hash(filename):
    h = hashlib.sha1()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def blob_hash(data):
    if isinstance(data, str):
        data = data.encode()
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


class ArtifactManager:

    def __init__(self, stores: StoreManager,
                 db: RunDBInterface = None,
                 out_path='',
                 calc_hash=True):
        self.out_path = out_path
        self.calc_hash = calc_hash

        self.data_stores = stores
        self.artifact_db = db
        self.input_artifacts = {}
        self.artifacts = {}

    def to_dict(self, status):
        artifacts = []
        for artifact in self.artifacts.values():
            if isinstance(artifact, dict):
                artifacts.append(artifact)
            else:
                artifacts.append(artifact.base_dict())
        status[run_keys.artifacts] = artifacts

    def log_artifact(self, execution, item, body=None, target_path='', src_path='',
                     tag='', viewer='', upload=True, labels=None):
        if isinstance(item, str):
            key = item
            item = Artifact(key, body)
        else:
            key = item.key
            target_path = target_path or item.target_path

        # find the target path from defaults and config
        item.viewer = viewer or item.viewer
        item.src_path = src_path or item.src_path
        if not target_path:
            target_path = uxjoin(self.out_path, item.src_path or key, execution.iteration)
        item.target_path = target_path
        item.tree = execution.tag
        if labels:
            if not item.labels:
                item.labels = {}
            for k, v in labels.items():
                item.labels[k] = str(v)

        self.artifacts[key] = item

        if upload:
            store, ipath = self.get_store(target_path)
            body = item.get_body()
            if body:
                if self.calc_hash:
                    item.hash = blob_hash(body)
                store.put(ipath, body)
            else:
                src_path = src_path or key
                if os.path.isfile(src_path):
                    if self.calc_hash:
                        item.hash = file_hash(src_path)
                    store.upload(ipath, src_path)

        if self.artifact_db:
            if not item.sources:
                item.sources = execution.to_dict()['spec'][run_keys.inputs]
            item.producer = execution.get_meta()
            if execution.iteration:
                key = '{}-{}'.format(execution.iteration, key)
                item.iter = execution.iteration
            self.artifact_db.store_artifact(key, item.to_dict(), item.tree,
                                            tag, execution.project)

    def get_store(self, url):
        return self.data_stores.get_or_create_store(url)


class Artifact(ModelObj):

    _dict_fields = ['key', 'kind', 'iter', 'tree', 'src_path', 'target_path', 'hash',
                    'description', 'viewer', 'inline', 'format']
    kind = ''

    def __init__(self, key, body=None, src_path=None, target_path='',
                 viewer=None, inline=False, format=None):
        self.key = key
        self.iter = None
        self.tree = None
        self.updated = None
        self.target_path = target_path
        self.src_path = src_path
        self._body = body
        self.format = format
        self.description = None
        self.viewer = viewer
        self.encoding = None
        self.labels = None
        self.annotations = None
        self.sources = []
        self.producer = None
        self.hash = None
        self._inline = inline
        self.license = ''
        self._post_init()

    def _post_init(self):
        pass

    @property
    def inline(self):
        if self._inline:
            return self.get_body()
        return None

    def get_body(self):
        return self._body

    def base_dict(self):
        return super().to_dict()

    def to_dict(self, fields=None):
        return super().to_dict(
            self._dict_fields + ['updated', 'labels', 'annotations', 'producer', 'sources'])


class ModelArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['framework']
    kind = 'model'

    def __init__(self, key, body=None, src_path=None, target_path='',
                 viewer=None, inline=False, format=None, framework=None):

        super().__init__(key, body, src_path, target_path, viewer, inline, format)
        self.framework = framework


class PlotArtifact(Artifact):
    kind = 'plot'
    def _post_init(self):
        self.viewer = 'chart'
        import matplotlib
        if not self._body or not isinstance(self._body, matplotlib.figure.Figure):
            raise ValueError('matplotlib fig must be provided as artifact body')
        if not pathlib.Path(self.key).suffix:
            self.key += '.html'

    def get_body(self):
        """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        canvas = FigureCanvas(self._body)
        png_output = BytesIO()
        canvas.print_png(png_output)
        data = png_output.getvalue()

        data_uri = base64.b64encode(data).decode('utf-8')
        return '<img title="{}" src="data:image/png;base64,{}">'.format(self.key, data_uri)


class TableArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['schema', 'header']
    kind = 'table'

    def __init__(self, key, body=None, df=None, src_path=None, target_path='',
                         viewer=None, visible=False, inline=False, format=None, header=None, schema=None):

        super().__init__(key, body, src_path, target_path, viewer, inline, format)
        key_suffix = pathlib.Path(key).suffix
        if not format and key_suffix:
            format = key_suffix[1:]

        if df is not None:
            self._is_df = True
            self.header = df.columns.values.tolist()
            format = format or 'csv'
            if format not in ['csv']:  # todo other formats
                raise ValueError('format must be csv for now')
            if visible and not key_suffix:
                key += '.csv'
            body = df
        else:
            self._is_df = False
            self.header = header

        self.format = format
        self.schema = schema
        if not viewer:
            viewer = 'table' if visible else None
        super().__init__(key, body, src_path, target_path, viewer, inline)

    def get_body(self):
        if not self._is_df:
            return self._body
        csv_buffer = StringIO()
        self._body.to_csv(csv_buffer, index=False, line_terminator='\n', encoding='utf-8')
        return csv_buffer.getvalue()


chart_template = '''
<html>
  <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable($data$);
        var options = $opts$;
        var chart = new google.visualization.$chart$(document.getElementById('chart_div'));
        chart.draw(data, options);
      }
    </script>
  </head>
  <body>
    <div id="chart_div" style="width: 100%; height: 500px;"></div>
  </body>
</html>
'''

class ChartArtifact(Artifact):
    kind = 'chart'
    _dict_fields = ['key', 'kind', 'iter', 'tree', 'src_path', 'target_path', 'hash',
                    'description', 'viewer']

    def __init__(self, key, data=[], src_path=None, target_path='',
                         viewer='chart', options={}):
        super().__init__(key, None, src_path, target_path, viewer)
        self.header = []
        self._rows = []
        if data:
            self.header = data[0]
            self._rows = data[1:]
        self.options = options
        self.chart = 'LineChart'

    def add_row(self, row):
        self._rows += [row]

    def get_body(self):
        if not self.options.get('title'):
            self.options['title'] = self.key
        data = [self.header] + self._rows
        return chart_template.replace('$data$', dict_to_json(data))\
            .replace('$opts$', dict_to_json(self.options))\
            .replace('$chart$', self.chart)

