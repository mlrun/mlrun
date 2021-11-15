import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from pandas.api.types import is_numeric_dtype, is_string_dtype

import mlrun

warnings.simplefilter(action="ignore", category=FutureWarning)


def gen_bool_list(output_df: pd.DataFrame, col: str) -> list:
    """
    Plotly requires a bool list for every column in order to determine what should be
    ploted (True) and what shouldn't (False).
    :param output_df: Result of the hyperparameter run as a Dataframe
    :param col: name of the col for which a bool list will be generated
    :returns bool_list: list of boolean corresponding to each column in out output df.
    """

    bool_list = []
    for i in output_df.columns:
        if col == i:
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list


def gen_dropdown_buttons(output_df: pd.DataFrame) -> list:
    """
    Uses each col name of the output dataframe to generate an equivalent dropdown button for plotting.
    :param output_df: Dataframe containg the output (columns with 'output.') of the hyperparameter run
    :returns buttons: list of dropdown buttons with their equivalent plot visibilty.
    """

    buttons = [
        dict(
            label=col,
            method="update",
            args=[{"visible": gen_bool_list(output_df, col)}],
        )
        for col in output_df.columns
    ]

    return buttons


def drop_exclusions(source_df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    """
    Ensures the param will be excluded if the user forgot to write 'param.' before the col name.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param exclude: User-provided list of parameters to be excluded from the graph
    :returns source_df: Our source_df without the columns the user whiches to drop.
    """

    for col in exclude:
        if col not in source_df.columns and f"param.{col}" in source_df.columns:
            source_df = source_df.drop(columns=[f"param.{col}"])

        elif col in source_df.columns:
            source_df = source_df.drop(columns=[col])

    return source_df


def gen_dimensions(df: pd.DataFrame, col: str) -> dict:
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

    if col == "iter":
        dimension["tickvals"] = (np.arange(1, max(df[col] + 1))).tolist()

    # Axis name
    dimension["label"] = col

    return dimension


def gen_plot_data(
    source_df: pd.DataFrame,
    param_df: pd.DataFrame,
    output_df: pd.DataFrame,
    colorscale: str,
) -> list:
    """
    Creates a list composed of the data to be plotted as a Parallel Coordinate, this includes
    the dimensions, dropdown buttons, and visibility of plots.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param param_df: Dataframe of parameters from the hyperparameter run
    :param output_df: Dataframe of outputs from the hyperparameter run
    :param colorscale: colors used for the lines in the parallel coordinate plot
    :returns data: list of information plotly requires to plot Parallel Coordinates.
    """

    data = []

    if "iter" in source_df.columns:
        index_col = "iter"
    elif "run" in source_df.columns:
        index_col = "run"

    for output in output_df.columns:
        # Creating Axes and Dimensions
        dimensions = [(gen_dimensions(param_df, col)) for col in param_df.columns]
        dimensions.insert(0, gen_dimensions(source_df, index_col))
        dimensions.append(gen_dimensions(output_df, output))

        # Plot Visibility on Dropdown
        visibility = True if output == output_df.columns[0] else False

        # Appending to list
        data.append(
            go.Parcoords(
                line=dict(color=source_df[index_col], colorscale=colorscale),
                dimensions=dimensions,
                visible=visibility,
            )
        )
    return data


def drop_identical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with identical values throughout iterations (no variance).
    :param df: a hyperparameter run as a Dataframe
    :returns df: Our Dataframe without identical-values.
    """

    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])
    return df


def split_dataframe(source_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """
    Splits the original hyperparameter dataframe into a params dataframe and result dataframe.
    :param source_df: Result of the hyperparameter run as a Dataframe

    :returns param_df: Dataframe of parameters from the hyperparameter run.
    :returns output_df: Dataframe of outputs from the hyperparameter run.
    """

    # Create param and output dataframes
    param_df = source_df.filter(like="param.")
    output_df = source_df.filter(like="output.")
    return param_df, output_df


def clean_col_names(df: pd.DataFrame, to_replace: str) -> pd.DataFrame:
    """
    Remove prefix of a column name within a Dataframe.
    :param df: the Dataframe containing the column names without prefix
    :param to_replace: the prefix to remove
    :returns: a cleaned dataframe.
    """
    df.columns = df.columns.str.replace(to_replace, "")
    return df


def compare_runs(
    run_name=None,
    project_name=None,
    runs_list=None,
    labels=None,
    iter=False,
    start_time_from: datetime = None,
    hide_identical: bool = True,
    exclude: list = ["label_column", "labels"],
    show=None,
    colorscale: str = "Blues",
    **kwargs,
) -> str:
    """
    Get the runs or project runs, creates param/output dataframe for each experiment and send the
    data to be plotted as parallel coordinates.

    :param run_name: Name of the run to retrieve
    :param project_name: Project that the runs belongs to
    :param runs_list: Run list object
    :param labels: List runs that have a specific label assigned. Currently only a single label filter can be
            applied, otherwise result will be empty.
    :param iter: If ``True`` return runs from all iterations. Otherwise, return only runs whose ``iter`` is 0.
    :param start_time_from: Filter by run start time in ``[start_time_from, start_time_to]``.
    :param hide_identical: Ignores parameters that remain the same throughout iterations
    :param exclude: User-provided list of parameters to be excluded from the graph
    :param show: Allows the user to display the plot within the notebook
    :param colorscale: colors used for the lines in the parallel coordinate plot
    :returns: param/output dataframes to be plotted
    """

    if runs_list is True:
        # Run list object
        runs_df = mlrun.lists.RunList(runs_list).to_df()

    elif runs_list is True and (run_name or project_name):
        raise Exception(
            "a list of runs and a project_name/run_name cannot both be passed"
        )

    else:
        runs_df = (
            mlrun.get_run_db()
            .list_runs(
                labels=labels,
                iter=iter,
                start_time_from=start_time_from,
                name=run_name,
                project=project_name,
                **kwargs,
            )
            .to_df()
        )

    # Remove empty param runs
    runs_df = runs_df[(runs_df["parameters"].str.len() >= 1)]

    # Remove empty output runs
    runs_df = runs_df[(runs_df["results"].str.len() >= 1)]

    # Create param dataframe and add prefix
    param_list = [p for p in runs_df["parameters"]]
    param_df = pd.DataFrame(param_list).add_prefix("param.")

    # Create output dataframe and add prefix
    output_list = [o for o in runs_df["results"]]
    output_df = pd.DataFrame(output_list).add_prefix("output.")

    return plot_parallel_coordinates(
        param_df=param_df,
        output_df=output_df,
        hide_identical=hide_identical,
        exclude=exclude,
        show=show,
        run_plot=True,
        colorscale=colorscale,
    )


def plot_parallel_coordinates(
    source_df: pd.DataFrame = pd.DataFrame(),
    param_df: pd.DataFrame = pd.DataFrame(),
    output_df: pd.DataFrame = pd.DataFrame(),
    hide_identical: bool = True,
    exclude: list = ["label_column", "labels"],
    show: bool = None,
    run_plot: bool = False,
    colorscale: str = "viridis",
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

    # users passes param_df and output_df
    if source_df.empty and param_df.empty is False and output_df.empty is False:
        source_df = pd.concat([param_df, output_df], axis=1, join="inner")
        if run_plot is True:
            source_df["run"] = range(1, 1 + len(param_df))
        else:
            source_df["iter"] = range(1, 1 + len(param_df))

    elif source_df.empty is False and param_df.empty and output_df.empty:
        param_df, output_df = split_dataframe(source_df)

    # Remove 'output.' and 'param.' from columns str
    param_df = clean_col_names(param_df, to_replace="param.")
    output_df = clean_col_names(output_df, to_replace="output.")

    # Drop unwanted columns
    param_df = drop_exclusions(param_df, exclude)
    output_df = drop_exclusions(output_df, exclude)

    # Drop identical columns
    if hide_identical:
        param_df = drop_identical(param_df)

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
                buttons=gen_dropdown_buttons(output_df),
            )
        ],
    )

    # Generate plotly figure with dropdown
    fig = go.Figure(
        data=gen_plot_data(source_df, param_df, output_df, colorscale), layout=layout
    )

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

    # Creating an html rendering of the plot
    plot_as_html = fig.to_html()

    if show or (show is None and mlrun.utils.is_ipython):
        display(HTML(plot_as_html))
    else:
        return plot_as_html
