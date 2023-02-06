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
#
from typing import Dict, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .model_monitoring_batch import DriftResultType, DriftStatus


class FeaturesDriftTablePlot:
    """
    Class for producing a features drift table. The plot is a table with columns of all the statistics and metrics
    provided with two additional plot columns of the histograms and drift notification. The rows content will be drawn
    per feature.

    For example, if the statistics are 'mean', 'min', 'max' and one metric of 'tvd', for 3 features the table will be:

    | feature name |       mean     |       min      |       max      |  tvd  |   histograms   |   |
    |______________|________________|________________|________________|_______|________________|___|
    |              | sample | input | sample | input | sample | input |       |                |   |
    |______________|________|_______|________|_______|________|_______|_______|________________|___|
    | feature 1    | 0.1    | 0.15  | 0.02   | 0.03  | 12.0   | 15.8  | 0.1   | ...            | V |
    | feature 2    | 0.1    | 0.15  | 0.02   | 0.03  | 12.0   | 15.8  | 0.1   | ...            | V |
    | feature 3    | 0.1    | 0.15  | 0.02   | 0.03  | 12.0   | 15.8  | 0.1   | ...            | V |
    """

    # Table column widths:
    _FEATURE_NAME_COLUMN_WIDTH = 140
    _VALUE_COLUMN_WIDTH = (
        70  # The width for the values of all the statistics and metrics columns.
    )
    _HISTOGRAMS_COLUMN_WIDTH = 180
    _NOTIFICATIONS_COLUMN_WIDTH = 20

    # Table rows heights:
    _HEADER_ROW_HEIGHT = 25
    _FEATURE_ROW_HEIGHT = 50  # Will be increased depends on the longest feature name.

    # Histograms configurations:
    _SAMPLE_SET_HISTOGRAM_COLOR = "rgb(0,112,192)"  # Blue
    _INPUTS_HISTOGRAM_COLOR = "rgb(208,0,106)"  # Magenta

    # Notification configurations:
    _NOTIFICATION_COLORS = {
        DriftStatus.NO_DRIFT: "rgb(0,176,80)",  # Green
        DriftStatus.POSSIBLE_DRIFT: "rgb(255,192,0)",  # Orange
        DriftStatus.DRIFT_DETECTED: "rgb(208,0,106)",  # Magenta
    }

    # Font configurations:
    _FONT_SIZE = 10
    _FONT_COLOR = "rgb(68,68,68)"  # Dark Grey

    # General configurations:
    _FEATURE_NAME_MAX_LENGTH = (
        16  # Each 16 characters the feature name will continue in a new line.
    )
    _VALUES_MAX_DIGITS = (
        6  # Each of the statistics and metrics will be rounded to show 6 digits.
    )
    _BACKGROUND_COLOR = "rgb(255,255,255)"  # White
    _SEPARATORS_COLOR = "rgb(240,240,240)"  # Light grey

    # File name:
    _FILE_NAME = "table_plot.html"

    def __init__(self):
        """
        Initialize the plot producer for later calling the `produce` method.
        """
        # Initialize columns names lists for the table that must be read before plotting:
        self._statistics_columns = None  # type: List[str]
        self._metrics_columns = None  # type: List[str]
        self._plot_columns = None  # type: List[str]
        self._value_columns_widths = None  # type: List[int]

    def produce(
        self,
        features: List[str],
        sample_set_statistics: dict,
        inputs_statistics: dict,
        metrics: Dict[str, Union[dict, float]],
        drift_results: Dict[str, DriftResultType],
    ) -> str:
        """
        Produce the html code of the table plot with the given information and the stored configurations in the class.

        :param features:              List of all the features names to include in the table. These names expected to be
                                      in the statistics and metrics dictionaries.
        :param sample_set_statistics: The sample set calculated statistics dictionary.
        :param inputs_statistics:     The inputs calculated statistics dictionary.
        :param metrics:               The drift detection metrics calculated on the sample set and inputs.
        :param drift_results:         The drift results per feature according to the rules of the monitor.

        :return: The full path to the html file of the plot.
        """
        # Plot the drift table:
        figure = self._plot(
            features=features,
            sample_set_statistics=sample_set_statistics,
            inputs_statistics=inputs_statistics,
            metrics=metrics,
            drift_results=drift_results,
        )

        # Get its HTML representation:
        figure_html = figure.to_html()

        # Turn off the table columns dragging by injecting the following JavaScript code:
        start, end = figure_html.rsplit(";", 1)
        middle = (
            ';for (const element of document.getElementsByClassName("table")) '
            '{element.style.pointerEvents = "none";}'
        )
        figure_html = start + middle + end

        return figure_html

    def _read_columns_names(self, statistics_dictionary: dict, drift_metrics: dict):
        """
        Read the available statistics and metrics to include them as columns in the table.

        :param statistics_dictionary: A statistics dictionary (one of sample_set or inputs).
        :param drift_metrics:         The metrics dictionary.
        """
        # Take all statistics from the first feature:
        self._statistics_columns = list(list(statistics_dictionary.values())[0].keys())
        self._statistics_columns.remove("hist")

        # Take all the metrics from the first feature:
        self._metrics_columns = list(list(drift_metrics.values())[0].keys())

        # Add the 'histograms' column if available:
        self._plot_columns = ["histograms", "detection"]

    def _calculate_columns_widths(self):
        """
        Calculate the column widths to draw the table. Can be called only after the method `_read_columns_names` was
        called.
        """
        # For the statistics, 2x`value_width` as there will be two values, one for the sample and one for the input:
        self._value_columns_widths = [2 * self._VALUE_COLUMN_WIDTH] * len(
            self._statistics_columns
        )

        # For the metrics:
        self._value_columns_widths += [self._VALUE_COLUMN_WIDTH] * len(
            self._metrics_columns
        )

    def _plot_headers_tables(self) -> Tuple[go.Table, go.Table]:
        """
        Plot the headers of the table:

        * The main header with the column names.
        * The sub-header with the statistics column splits to 'sample | inputs'.

        :return: A tuple with both headers - `Table` traces:
                 [0] - Header.
                 [1] - Sub-header.
        """
        # Generate the header of the table:
        headers = [
            # [0] - The feature name column
            "",
            # [1:-2] - Statistics and metrics columns.
            *self._statistics_columns,
            *self._metrics_columns,
            # [-2] - The histograms column
            "histograms",
            # [-1] - The notifications column
            "",
        ]
        header_table = go.Table(
            header={
                "values": [
                    f"<b>{header.capitalize()}</b>" for header in headers
                ],  # Make the text bold and with starting capital letter.
                "align": "center",
                "font": {"size": self._FONT_SIZE, "color": self._FONT_COLOR},
                "line": {"color": self._SEPARATORS_COLOR},
            },
            columnwidth=[
                self._FEATURE_NAME_COLUMN_WIDTH,
                *self._value_columns_widths,
                self._HISTOGRAMS_COLUMN_WIDTH,
                self._NOTIFICATIONS_COLUMN_WIDTH,
            ],
            header_fill_color=self._BACKGROUND_COLOR,
        )

        # Generate the sub-headers (for each of the statistics column there should be two columns, one for the sample
        # set and one for the inputs):
        sub_headers = (
            [""]
            + ["Sample", "Input"] * len(self._statistics_columns)
            + [""] * len(self._metrics_columns)
            + ["", ""]
        )
        sub_header_table = go.Table(
            header={
                "values": sub_headers,
                "align": "center",
                "font": {"size": self._FONT_SIZE, "color": self._FONT_COLOR},
                "line": {"color": self._SEPARATORS_COLOR},
            },
            columnwidth=(
                [self._FEATURE_NAME_COLUMN_WIDTH]
                + [self._VALUE_COLUMN_WIDTH]
                * (2 * len(self._statistics_columns) + len(self._metrics_columns))
                + [self._HISTOGRAMS_COLUMN_WIDTH, self._NOTIFICATIONS_COLUMN_WIDTH]
            ),
            header_fill_color=self._BACKGROUND_COLOR,
        )

        return header_table, sub_header_table

    def _separate_feature_name(self, feature_name: str) -> List[str]:
        """
        Separate the given feature name by the maximum length configured in the class. Used for calculating the amount
        of lines required to represent the longest feature name in the table, so the row heights will fit accordingly.

        :param feature_name: The feature name to separate.

        :return: The feature name's list of the separations.
        """
        return [
            feature_name[i : i + self._FEATURE_NAME_MAX_LENGTH]
            for i in range(0, len(feature_name), self._FEATURE_NAME_MAX_LENGTH)
        ]

    def _get_value_format(self, value: Union[str, int, float]) -> str:
        """
        Plotly uses D3 formatter to format values. This method return the format according to the configured
        properties in the class to the given value - one of the cells in a feature row statistics and metrics values.

        :param value: A value of a feature row cell.

        :return: The value's format.
        """
        # A string does not need reformatting:
        if isinstance(value, str):
            return ""

        # Check fo nan values:
        if np.isnan(value):
            return ""

        # Any whole number or number with a long integer value should be parsed into short characters
        # (e.g: 10000 -> 10k, 1100000 -> 1.1m):
        integer_length = len(str(int(value)))
        if value % 1 == 0 or integer_length > self._VALUES_MAX_DIGITS:
            return "~s"

        # Round it to the remaining digits left considering the integer length:
        return f".{self._VALUES_MAX_DIGITS - integer_length}"

    def _plot_feature_row_table(
        self,
        feature_name: str,
        sample_statistics: dict,
        input_statistics: dict,
        metrics: dict,
        row_height: int,
    ) -> go.Table:
        """
        Plot the feature row to include in the table. The row will include only the columns of the statistics and
        metrics values. The histogram and drift notification are plotted in different methods.

        :param feature_name:      The feature's name.
        :param sample_statistics: The feature's sample set statistics dictionary.
        :param input_statistics:  The feature's inputs statistics dictionary.
        :param metrics:           The feature's metrics results dictionary.
        :param row_height:        The height of the row to plot.

        :return: The feature row - `Table` trace.
        """
        # Add '\n' to the feature name in order to make it fit into its cell:
        feature_name = "<br>".join(self._separate_feature_name(feature_name))

        # Initialize the cells values list with the bold feature name as the first value:
        cells_values = [f"<b>{feature_name}</b>"]

        # Add the statistics columns:
        for column in self._statistics_columns:
            cells_values.append(sample_statistics[column])
            cells_values.append(input_statistics[column])

        # Add the metrics columns:
        for column in self._metrics_columns:
            cells_values.append(metrics[column])

        # Get the cells values formats:
        cells_formats = [
            self._get_value_format(value=cell_value) for cell_value in cells_values
        ]

        # Create the row:
        feature_row_table = go.Table(
            header={
                "values": cells_values,
                "align": "center",
                "font": {"size": self._FONT_SIZE, "color": self._FONT_COLOR},
                "line": {"color": self._SEPARATORS_COLOR},
                "height": row_height,
                "format": cells_formats,
            },
            columnwidth=[self._FEATURE_NAME_COLUMN_WIDTH, self._VALUE_COLUMN_WIDTH],
            header_fill_color=self._BACKGROUND_COLOR,
        )

        return feature_row_table

    def _plot_histogram_scatters(
        self, sample_hist: Tuple[list, list], input_hist: Tuple[list, list]
    ) -> Tuple[go.Scatter, go.Scatter]:
        """
        Plot the feature's histograms to include in the "histograms" column. Both histograms are returned to later be
        added in the same figure, so they will be on top of each other and not separated. Both histograms are rescaled
        to be from 0.0 to 1.0, so they will be drawn in the same scale regardless the amount of elements they were
        calculated upon.

        :param sample_hist: The sample set histogram data.
        :param input_hist:  The input histogram data.

        :return: A tuple with both histograms - `Scatter` traces:
                 [0] - Sample set histogram.
                 [1] - Input histogram.
        """
        # Initialize a list to collect the scatters:
        scatters = []

        # Plot the histograms:
        for name, color, histogram in zip(
            ["sample", "input"],
            [self._SAMPLE_SET_HISTOGRAM_COLOR, self._INPUTS_HISTOGRAM_COLOR],
            [sample_hist, input_hist],
        ):
            # Read the histogram tuple:
            counts, bins = histogram
            # Rescale the counts to be in percentages (between 0.0 to 1.0):
            counts = np.array(counts) / sum(counts)
            # Convert to NumPy for vectorization:
            bins = np.array(bins)
            # Center the bins (leave the first one):
            bins = 0.5 * (bins[:-1] + bins[1:])
            # Plot the histogram as a line with filled background below it:
            histogram_scatter = go.Scatter(
                x=bins,
                y=counts,
                fill="tozeroy",
                name=name,
                line_shape="spline",  # Make the line rounder.
                line={"color": color},
                legendgroup=name,
            )
            scatters.append(histogram_scatter)

        return scatters[0], scatters[1]

    def _calculate_row_height(self, features: List[str]) -> int:
        """
        Calculate the feature row height according to the given features. The longest feature will set the height to all
        the rows. The height depends on the separations amount of the longest feature name - more '\n' means more pixels
        for the row's height to show the entirety of the feature name.

        :param features: The list of features.

        :return: The row height.
        """
        feature_name_seperations = max(
            [
                len(self._separate_feature_name(feature_name))
                for feature_name in features
            ]
        )
        return max(
            self._FEATURE_ROW_HEIGHT, 1.5 * self._FONT_SIZE * feature_name_seperations
        )

    def _plot_notification_circle(
        self,
        figure: go.Figure,
        row: int,
        row_height: int,
        drift_result: DriftResultType,
    ):
        """
        Plot the drift notification - a little circle with color as configured in the class. The color will beb chosen
        according to the drift status given.

        :param figure:       The figure (feature row cell) to draw the circle in.
        :param row:          The row number.
        :param row_height:   The row height.
        :param drift_result: The drift result.
        """
        # Update the x-axis and y-axis to be in range of the `row_height` x `self._NOTIFICATIONS_COLUMN_WIDTH` so the
        # circle will be sized correctly. The axis number is equal to the row number - 3 (plots with axes starts at
        # row 3) times the plot columns (2 columns has axes in each row) + 2 (to get to the column of the notification):
        axis_number = (row - 3) * 2 + 2
        figure["layout"][f"xaxis{axis_number}"].update(
            range=[0, self._NOTIFICATIONS_COLUMN_WIDTH]
        )
        figure["layout"][f"yaxis{axis_number}"].update(range=[0, row_height])

        # Get the color:
        notification_color = self._NOTIFICATION_COLORS[drift_result[0]]
        half_transparent_notification_color = notification_color.replace(
            "rgb", "rgba"
        ).replace(")", ",0.5)")

        # Calculate the circle points adjusted to the given row height (aimed to where the text starts and to be at the
        # size of the text as well):
        y0 = 36 + (row_height - self._FEATURE_ROW_HEIGHT)
        y1 = y0 + self._FONT_SIZE
        x0 = (self._NOTIFICATIONS_COLUMN_WIDTH / 2) - ((y1 - y0) / 2)
        x1 = (self._NOTIFICATIONS_COLUMN_WIDTH / 2) + ((y1 - y0) / 2)

        # Draw the circle on top of the figure:
        figure.add_shape(
            type="circle",
            xref="x",
            yref="y",
            fillcolor=half_transparent_notification_color,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line_color=notification_color,
            row=row,
            col=3,
        )

    def _plot(
        self,
        features: List[str],
        sample_set_statistics: dict,
        inputs_statistics: dict,
        metrics: Dict[str, Union[dict, float]],
        drift_results: Dict[str, DriftResultType],
    ) -> go.Figure:
        """
        Plot the drift table using the given data and stored configurations of the class.

        :param features:              List of all the features names to include in the table. These names expected to be
                                      in the statistics and metrics dictionaries.
        :param sample_set_statistics: The sample set calculated statistics dictionary.
        :param inputs_statistics:     The inputs calculated statistics dictionary.
        :param metrics:               The drift detection metrics calculated on the sample set and inputs.
        :param drift_results:         The drift results per feature according to the rules of the monitor.

        :return: The plot's figure.
        """
        # Read and calculate widths and heights for the table:
        self._read_columns_names(
            statistics_dictionary=sample_set_statistics, drift_metrics=metrics
        )
        self._calculate_columns_widths()
        row_height = self._calculate_row_height(features=features)
        rows = 2 + len(features)  # Header + sub-header + row per feature
        columns = 3  # Values table + histograms plot + detection plot

        # Calculate the height and width of the entire plot:
        width = (
            self._FEATURE_NAME_COLUMN_WIDTH
            + sum(self._value_columns_widths)
            + self._HISTOGRAMS_COLUMN_WIDTH
            + self._NOTIFICATIONS_COLUMN_WIDTH
        )
        height = 2 * self._HEADER_ROW_HEIGHT + len(features) * row_height

        # Create the main figure - a subplots figure with the calculated rows and columns (the `row_heights` and
        # `column_widths` attributes here are calculated in percentages):
        main_figure = make_subplots(
            rows=rows,
            cols=columns,
            specs=(
                [[{"type": "table", "colspan": 3}, None, None]] * 2
                + [[{"type": "table"}, {}, {}]] * (rows - 2)
            ),
            row_heights=(
                [self._HEADER_ROW_HEIGHT / height] * 2
                + [row_height / height] * (rows - 2)
            ),
            column_widths=[
                (self._FEATURE_NAME_COLUMN_WIDTH + sum(self._value_columns_widths))
                / width,
                self._HISTOGRAMS_COLUMN_WIDTH / width,
                self._NOTIFICATIONS_COLUMN_WIDTH / width,
            ],
            horizontal_spacing=0,
            vertical_spacing=0,
        )

        # Add the first two rows - the header and sub-header:
        header_trace, sub_header_trace = self._plot_headers_tables()
        main_figure.add_trace(header_trace, row=1, col=1)
        main_figure.add_trace(sub_header_trace, row=2, col=1)

        # Start going over the features and plot each row, histogram and notification:
        row = 3  # We are currently at row 3 counting the headers.
        for feature in features:
            # Add the feature values:
            main_figure.add_trace(
                self._plot_feature_row_table(
                    feature_name=feature,
                    sample_statistics=sample_set_statistics[feature],
                    input_statistics=inputs_statistics[feature],
                    metrics=metrics[feature],
                    row_height=row_height,
                ),
                row=row,
                col=1,
            )
            # Add the histograms (both traces are added to the same subplot figure):
            sample_hist, input_hist = self._plot_histogram_scatters(
                sample_hist=sample_set_statistics[feature]["hist"],
                input_hist=inputs_statistics[feature]["hist"],
            )
            if row != 3:  # Only the first row should have its legend visible:
                sample_hist.showlegend = False
                input_hist.showlegend = False
            main_figure.add_trace(sample_hist, row=row, col=2)
            main_figure.add_trace(input_hist, row=row, col=2)
            # Add the notification (a circle with color according to the drift alert):
            self._plot_notification_circle(
                figure=main_figure,
                row=row,
                row_height=row_height,
                drift_result=drift_results[feature],
            )
            row += 1

        # Configure the layout and axes for height and widths:
        main_figure.update_layout(
            autosize=False,
            width=width,
            height=height,
            plot_bgcolor=self._BACKGROUND_COLOR,
            paper_bgcolor=self._BACKGROUND_COLOR,
            margin={"t": 0, "b": 0, "l": 0, "r": 0, "autoexpand": False},
            font_size=self._FONT_SIZE,  # Control the font of the x and y axes and the legend.
            font_color=self._FONT_COLOR,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": 1.0 - (self._HEADER_ROW_HEIGHT / height) + 0.002,
                "xanchor": "right",
                "x": 1.0 - (self._NOTIFICATIONS_COLUMN_WIDTH / width) - 0.01,
                "bgcolor": "rgba(0,0,0,0)",
            },
        )
        main_figure.update_xaxes(
            showticklabels=False,
            showline=True,
            linewidth=1,
            linecolor=self._SEPARATORS_COLOR,
            mirror=True,
        )
        main_figure.update_yaxes(
            showticklabels=False,
            showline=True,
            linewidth=1,
            linecolor=self._SEPARATORS_COLOR,
            mirror=True,
        )

        return main_figure
