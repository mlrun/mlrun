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
import base64
from io import BytesIO

from ..utils import dict_to_json
from .base import Artifact


class PlotArtifact(Artifact):
    _TEMPLATE = """
<h3 style="text-align:center">{}</h3>
<img title="{}" src="data:image/png;base64,{}">
"""
    kind = "plot"

    def __init__(
        self, key=None, body=None, is_inline=False, target_path=None, title=None
    ):
        super().__init__(key, body, format="html", target_path=target_path)
        self.description = title

    def before_log(self):
        self.viewer = "chart"
        import matplotlib

        if not self._body or not isinstance(
            self._body, (bytes, matplotlib.figure.Figure)
        ):
            raise ValueError(
                "matplotlib fig or png bytes must be provided as artifact body"
            )

    def get_body(self):
        """Convert Matplotlib figure 'fig' into a <img> tag for HTML use
        using base64 encoding."""
        if isinstance(self._body, bytes):
            data = self._body
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            canvas = FigureCanvas(self._body)
            png_output = BytesIO()
            canvas.print_png(png_output)
            data = png_output.getvalue()

        data_uri = base64.b64encode(data).decode("utf-8")
        return self._TEMPLATE.format(self.description or self.key, self.key, data_uri)


class ChartArtifact(Artifact):
    _TEMPLATE = """
<html>
  <head>
    <script
        type="text/javascript"
        src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable($data$);
        var options = $opts$;
        var chart = new google.visualization.$chart$(
            document.getElementById('chart_div'));
        chart.draw(data, options);
      }
    </script>
  </head>
  <body>
    <div id="chart_div" style="width: 100%; height: 500px;"></div>
  </body>
</html>
"""

    kind = "chart"

    def __init__(
        self,
        key=None,
        data=None,
        header=None,
        options=None,
        title=None,
        chart=None,
        target_path=None,
    ):
        data = [] if data is None else data
        options = {} if options is None else options
        super().__init__(key, target_path=target_path)
        self.viewer = "chart"
        self.header = header or []
        self.title = title
        self.rows = []
        if data:
            if header:
                self.rows = data
            else:
                self.header = data[0]
                self.rows = data[1:]
        self.options = options
        self.chart = chart or "LineChart"
        self.format = "html"

    def add_row(self, row):
        self.rows += [row]

    def get_body(self):
        if not self.options.get("title"):
            self.options["title"] = self.title or self.key
        data = [self.header] + self.rows
        return (
            self._TEMPLATE.replace("$data$", dict_to_json(data))
            .replace("$opts$", dict_to_json(self.options))
            .replace("$chart$", self.chart)
        )


class BokehArtifact(Artifact):
    """
    Bokeh artifact is an artifact for saving Bokeh generated figures. They will be stored in a html format.
    """

    kind = "bokeh"

    def __init__(
        self, figure, key: str = None, target_path: str = None,
    ):
        """
        Initialize a Bokeh artifact with the given figure.

        :param figure:      Bokeh figure ('bokeh.plotting.Figure' object) to save as an artifact.
        :param key:         Key for the artifact to be stored in the database.
        :param target_path: Path to save the artifact.
        """
        super().__init__(key=key, target_path=target_path, viewer="bokeh")

        # Validate input:
        try:
            from bokeh.plotting import Figure
        except (ModuleNotFoundError, ImportError) as Error:
            raise Error(
                "Using 'BokehArtifact' requires bokeh package. Use pip install mlrun[bokeh] to install it."
            )
        if not isinstance(figure, Figure):
            raise ValueError(
                "BokehArtifact requires the figure parameter to be a "
                "'bokeh.plotting.Figure' but received '{}'".format(type(figure))
            )

        # Continue initializing the bokeh artifact:
        self._figure = figure
        self.format = "html"

    def get_body(self):
        """
        Get the artifact's body - the bokeh figure's html code.

        :return: The figure's html code.
        """
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        return file_html(self._figure, CDN, self.key)


class PlotlyArtifact(Artifact):
    """
    Plotly artifact is an artifact for saving Plotly generated figures. They will be stored in a html format.
    """

    kind = "plotly"

    def __init__(
        self, figure, key: str = None, target_path: str = None,
    ):
        """
        Initialize a Plotly artifact with the given figure.

        :param figure:      Plotly figure ('plotly.graph_objs.Figure' object) to save as an artifact.
        :param key:         Key for the artifact to be stored in the database.
        :param target_path: Path to save the artifact.
        """
        super().__init__(key=key, target_path=target_path, viewer="plotly")

        # Validate input:
        try:
            from plotly.graph_objs import Figure
        except (ModuleNotFoundError, ImportError) as Error:
            raise Error(
                "Using 'PlotlyArtifact' requires plotly package. Use pip install mlrun[plotly] to install it."
            )
        if not isinstance(figure, Figure):
            raise ValueError(
                "PlotlyArtifact requires the figure parameter to be a "
                "'plotly.graph_objs.Figure' but received '{}'".format(type(figure))
            )

        # Continue initializing the plotly artifact:
        self._figure = figure
        self.format = "html"

    def get_body(self):
        """
        Get the artifact's body - the Plotly figure's html code.

        :return: The figure's html code.
        """
        return self._figure.write_html(self.key)
