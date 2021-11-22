import datetime
import warnings
from typing import List

import numpy as np
import pandas as pd
from IPython.core.display import HTML, display
from pandas.api.types import is_numeric_dtype, is_string_dtype

import mlrun
from mlrun.utils import flatten

warnings.simplefilter(action="ignore", category=FutureWarning)


def _gen_dropdown_buttons(output_cols) -> list:
    """Uses each output col name to generate an equivalent dropdown button for plotting."""

    output_cols = [col[len("output.") :] for col in output_cols]

    def gen_bool_list(col):
        return [name == col for name in output_cols]

    buttons = [
        dict(label=col, method="update", args=[{"visible": gen_bool_list(col)}],)
        for col in output_cols
    ]

    return buttons


def _gen_dimensions(
    df: pd.DataFrame, col: str, prefix: str = None, is_index=False
) -> dict:
    """
    Computes the plotting dimensions of each parameter/output col according to its type.
    :param df: Dataframe containing the data to be plotted
    :param col: Column name for which its dimensions will be computed
    :returns dimension: Dimensions and instructions required to plot a parameter.
    """

    dimension = {}

    if is_numeric_dtype(df[col]):
        dimension["range"] = [min(df[col]), max(df[col])]
        dimension["values"] = df[col]

    elif is_string_dtype(df[col]):
        dimension["range"] = [0, len(df[col])]
        dimension["values"] = [(list(df[col]).index(val)) for val in list(df[col])]
        dimension["tickvals"] = [(list(df[col]).index(val)) for val in list(df[col])]
        dimension["ticktext"] = list(df[col])

    if col == "iter" or is_index:
        dimension["tickvals"] = (np.arange(1, max(df[col] + 1))).tolist()

    # Axis name
    if prefix:
        col = col[len(prefix) :]
    dimension["label"] = col

    return dimension


def _drop_identical(df: pd.DataFrame, param_cols) -> pd.DataFrame:
    """Drop columns with identical values throughout iterations (no variance)."""

    dropped_cols = []
    for col in param_cols:
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])
            dropped_cols.append(col)
    return df, [col for col in param_cols if col not in dropped_cols]


def _get_column_names(df: pd.DataFrame):
    params = []
    outputs = []
    for name in df.columns:
        if name.startswith("param."):
            params.append(name)
        if name.startswith("output."):
            outputs.append(name)
    return params, outputs


def gen_pcp_plot(
    source_df: pd.DataFrame,
    index_col: str,
    hide_identical: bool = True,
    exclude: list = None,
    colorscale: str = None,
):
    """
    Creates a list composed of the data to be plotted as a Parallel Coordinate, this includes
    the dimensions, dropdown buttons, and visibility of plots.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param index_col: index column name
    :param hide_identical: Ignores parameters that remain the same throughout iterations
    :param exclude: User-provided list of parameters to be excluded from the graph
    :param colorscale: colors used for the lines in the parallel coordinate plot
    :returns plot_as_html: The Parallel Coordinate plot in HTML format.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(f"{exc}, please run 'pip install plotly'")

    data = []

    # Remove unwanted columns
    for col in exclude or []:
        if f"param.{col}" in source_df.columns:
            source_df = source_df.drop(columns=[f"param.{col}"])

    param_cols, output_cols = _get_column_names(source_df)

    # Drop identical columns
    if hide_identical:
        source_df, param_cols = _drop_identical(source_df, param_cols)

    for output in output_cols:
        # Creating Axes and Dimensions
        dimensions = [(_gen_dimensions(source_df, col, "param.")) for col in param_cols]
        dimensions.insert(0, _gen_dimensions(source_df, index_col))
        dimensions.append(_gen_dimensions(source_df, output, "output."))

        # Plot Visibility on Dropdown
        visibility = True if output == output_cols[0] else False

        # Appending to list
        data.append(
            go.Parcoords(
                line=dict(
                    color=source_df[index_col], colorscale=colorscale or "viridis"
                ),
                dimensions=dimensions,
                visible=visibility,
            )
        )

    # Set position of dropdown buttons
    layout = dict(
        showlegend=True,
        updatemenus=[
            dict(
                active=0,
                xanchor="left",
                x=0.10,
                yanchor="bottom",
                y=1.3,
                bgcolor="#ffffff",
                showactive=True,
                buttons=_gen_dropdown_buttons(output_cols),
            )
        ],
    )

    fig = go.Figure(data=data, layout=layout)

    # Workaround to remove dropdown button opacity
    fig.update_layout(autosize=True, margin=dict(l=50, r=50, b=50, t=50, pad=4))

    # Add annotation to dropdown
    fig.update_layout(
        annotations=[
            dict(
                text="output metric:",
                x=0.0,
                xref="paper",
                y=1.4,
                yref="paper",
                align="right",
                showarrow=False,
            ),
        ]
    )

    return fig.to_html()


def _show_and_export_html(html: str, show=None, filename=None, runs_list=None):
    if runs_list:
        html_table = runs_list.show(False, short=True)

    if filename:
        with open(filename, "w") as fp:
            if runs_list:
                fp.write(html[: html.rfind("</body>")])
                fp.write("<br>" + html_table)
                fp.write("</body></html>")
            else:
                fp.write(html)
    if show or (show is None and mlrun.utils.is_ipython):
        display(HTML(html))
        if runs_list:
            display(HTML(html_table))
    else:
        return html


def _runs_list_to_df(runs_list):
    runs_df = runs_list.to_df()

    # Remove empty param/result runs
    runs_df = runs_df[(runs_df["parameters"].str.len() >= 1)]
    runs_df = runs_df[(runs_df["results"].str.len() >= 1)]

    runs_df = flatten(runs_df, "parameters", "param.")
    runs_df = flatten(runs_df, "results", "output.")
    runs_df["iter"] = range(1, 1 + len(runs_df))
    return runs_df


def plot_parallel_coordinates(
    source_df: pd.DataFrame,
    hide_identical: bool = True,
    exclude: list = None,
    show: bool = None,
    colorscale: str = "viridis",
    index_col=None,
    filename=None,
) -> str:
    """
    Plots the output of the hyperparameter run in a Parallel Coordinate format using the Plotly library.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param param_df: Dataframe of parameters from the hyperparameter run
    :param output_df: Dataframe of outputs from the hyperparameter run
    :param hide_identical: Ignores parameters that remain the same throughout iterations
    :param exclude: User-provided list of parameters to be excluded from the graph
    :param show: Allows the user to display the plot within the notebook
    :param run_plot: Flag used if the data sources from runs or iterations
    :param colorscale: colors used for the lines in the parallel coordinate plot
    :returns plot_as_html: The Parallel Coordinate plot in HTML format.
    """

    # Generate plotly figure with dropdown
    plot_as_html = gen_pcp_plot(
        source_df,
        index_col=index_col or "iter",
        hide_identical=hide_identical,
        exclude=exclude,
        colorscale=colorscale,
    )

    if filename:
        with open(filename, "w") as fp:
            fp.write(plot_as_html)
    if show or (show is None and mlrun.utils.is_ipython):
        display(HTML(plot_as_html))
    else:
        return plot_as_html


def compare_run_iterations(
    run: mlrun.model.RunObject,
    hide_identical: bool = True,
    exclude: list = None,
    show: bool = None,
    filename=None,
    colorscale: str = None,
):
    """parallel coordinates plot to compare between run iterations (hyper-params)

    :param run:            MLRun run object
    :param hide_identical: hide columns with identical values
    :param exclude:        User-provided list of parameters to be excluded from the plot
    :param show:           Allows the user to display the plot within the notebook
    :param filename:       Output filename to save the plot html file
    :param colorscale:     colors used for the lines in the parallel coordinate plot
    :return:  plot html
    """

    source_df = pd.DataFrame(
        run.status.iterations[1:], columns=run.status.iterations[0]
    )
    plot_as_html = gen_pcp_plot(
        source_df,
        index_col="iter",
        hide_identical=hide_identical,
        exclude=exclude,
        colorscale=colorscale,
    )
    return _show_and_export_html(plot_as_html, show, filename)


def compare_runs_list(
    runs_list: List[mlrun.model.RunObject],
    hide_identical: bool = True,
    exclude: list = None,
    show: bool = None,
    filename=None,
    colorscale: str = None,
):
    """parallel coordinates plot to compare between a list of runs

    :param runs_list:      List of MLRun run object
    :param hide_identical: hide columns with identical values
    :param exclude:        User-provided list of parameters to be excluded from the plot
    :param show:           Allows the user to display the plot within the notebook
    :param filename:       Output filename to save the plot html file
    :param colorscale:     colors used for the lines in the parallel coordinate plot
    :return:  plot html
    """

    if isinstance(runs_list, list):
        runs_list = mlrun.lists.RunList([run.to_dict() for run in runs_list])
    source_df = _runs_list_to_df(runs_list)
    plot_as_html = gen_pcp_plot(
        source_df,
        index_col="iter",
        hide_identical=hide_identical,
        exclude=exclude,
        colorscale=colorscale,
    )
    return _show_and_export_html(plot_as_html, show, filename, runs_list=runs_list)


def compare_db_runs(
    run_name=None,
    project_name=None,
    labels=None,
    iter=False,
    start_time_from: datetime = None,
    hide_identical: bool = True,
    exclude: list = [],
    show=None,
    colorscale: str = "Blues",
    filename=None,
    **query_args,
) -> str:
    """
    Get the runs or project runs, creates param/output dataframe for each experiment and send the
    data to be plotted as parallel coordinates.

    :param run_name: Name of the run to retrieve
    :param project_name: Project that the runs belongs to
    :param labels: List runs that have a specific label assigned. Currently only a single label filter can be
            applied, otherwise result will be empty.
    :param iter:            If ``True`` return runs from all iterations. Otherwise, return only parent runs (iter 0).
    :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
    :param hide_identical:  hide columns with identical values
    :param exclude:         User-provided list of parameters to be excluded from the plot
    :param show:            Allows the user to display the plot within the notebook
    :param filename:        Output filename to save the plot html file
    :param colorscale:      colors used for the lines in the parallel coordinate plot
    :param query_args:      additional list_runs() query arguments
    :return:  plot html
    """

    runs_list = mlrun.get_run_db().list_runs(
        labels=labels,
        iter=iter,
        start_time_from=start_time_from,
        name=run_name,
        project=project_name,
        **query_args,
    )

    runs_df = _runs_list_to_df(runs_list)
    plot_as_html = gen_pcp_plot(
        runs_df,
        index_col="iter",
        hide_identical=hide_identical,
        exclude=exclude,
        colorscale=colorscale,
    )
    return _show_and_export_html(plot_as_html, show, filename, runs_list=runs_list)
