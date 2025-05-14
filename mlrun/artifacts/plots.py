# Copyright 2023 Iguazio
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
import typing
import warnings
from io import BytesIO

import mlrun

from .base import Artifact

if typing.TYPE_CHECKING:
    from plotly.graph_objs import Figure


class PlotArtifact(Artifact):
    kind = "plot"

    _TEMPLATE = """
<h3 style="text-align:center">{}</h3>
<img title="{}" src="data:image/png;base64,{}">
"""

    def __init__(
        self, key=None, body=None, is_inline=False, target_path=None, title=None
    ):
        if key or body or is_inline or target_path:
            warnings.warn(
                "Artifact constructor parameters are deprecated in 1.7.0 and will be removed in 1.10.0. "
                "Use the metadata and spec parameters instead.",
                DeprecationWarning,
            )
        super().__init__(key, body, format="html", target_path=target_path)
        self.metadata.description = title

    def before_log(self):
        self.spec.viewer = "chart"
        import matplotlib

        if not self.spec.get_body() or not isinstance(
            self.spec.get_body(), (bytes, matplotlib.figure.Figure)
        ):
            raise ValueError(
                "matplotlib fig or png bytes must be provided as artifact body"
            )

    def get_body(self):
        """Convert Matplotlib figure 'fig' into a <img> tag for HTML use
        using base64 encoding."""
        if isinstance(self.spec.get_body(), bytes):
            data = self.spec.get_body()
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            canvas = FigureCanvas(self.spec.get_body())
            png_output = BytesIO()
            canvas.print_png(png_output)
            data = png_output.getvalue()

        data_uri = base64.b64encode(data).decode("utf-8")
        return self._TEMPLATE.format(
            self.metadata.description or self.metadata.key, self.metadata.key, data_uri
        )


class PlotlyArtifact(Artifact):
    """
    Plotly artifact is an artifact for saving Plotly generated figures. They will be stored in a html format.
    """

    kind = "plotly"

    def __init__(
        self,
        figure: typing.Optional["Figure"] = None,
        key: typing.Optional[str] = None,
        target_path: typing.Optional[str] = None,
    ) -> None:
        """
        Initialize a Plotly artifact with the given figure.

        :param figure:      Plotly figure ('plotly.graph_objs.Figure' object) to save as an artifact.
        :param key:         Key for the artifact to be stored in the database.
        :param target_path: Path to save the artifact.
        """
        if key or target_path:
            warnings.warn(
                "Artifact constructor parameters are deprecated in 1.7.0 and will be removed in 1.10.0. "
                "Use the metadata and spec parameters instead.",
                DeprecationWarning,
            )
        # Validate the plotly package:
        try:
            from plotly.graph_objs import Figure
        except ModuleNotFoundError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "Using `PlotlyArtifact` requires plotly package. Use `pip install mlrun[plotly]` to install it."
            )
        except ImportError:
            import plotly

            raise mlrun.errors.MLRunMissingDependencyError(
                f"Using `PlotlyArtifact` requires plotly version >= 5.4.0 but found version {plotly.__version__}. "
                f"Use `pip install -U mlrun[plotly]` to install it."
            )

        # Call the artifact initializer:
        super().__init__(key=key, target_path=target_path, viewer="plotly")

        # Validate input:
        if figure is not None and not isinstance(figure, Figure):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"PlotlyArtifact requires the figure parameter to be a "
                f"`plotly.graph_objs.Figure` but received '{type(figure)}'"
            )

        # Continue initializing the plotly artifact:
        self._figure = figure
        self.spec.format = "html"

    def get_body(self) -> str:
        """
        Get the artifact's body - the Plotly figure's html code.

        :return: The figure's html code.
        """
        return self._figure.to_html()
