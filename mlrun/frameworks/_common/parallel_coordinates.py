import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from pandas.api.types import is_numeric_dtype, is_string_dtype


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

    buttons = []

    for col in output_df.columns:
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[{"visible": gen_bool_list(output_df, col)}],
            )
        )

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

        if col == "iter":
            dimension["tickvals"] = (np.arange(1, max(df[col] + 1))).tolist()

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
    source_df: pd.DataFrame, param_df: pd.DataFrame, output_df: pd.DataFrame
) -> list:
    """
    Creates a list composed of the data to be plotted as a Parallel Coordinate, this includes
    the dimensions, dropdown buttons, and visibility of plots.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param param_df: Dataframe of parameters from the hyperparameter run
    :param output_df: Dataframe of outputs from the hyperparameter run
    :returns data: list of information plotly requires to plot Parallel Coordinates.
    """

    data = []

    for output in output_df.columns:
        # Creating Axes and Dimensions
        dimensions = [(gen_dimensions(param_df, col)) for col in param_df.columns]
        dimensions.insert(0, gen_dimensions(source_df, "iter"))
        dimensions.append(gen_dimensions(output_df, output))

        # Plot Visibility on Dropdown
        visibility = True if output == output_df.columns[0] else False

        # Appending to list
        data.append(
            go.Parcoords(
                line=dict(color=source_df["iter"], colorscale="viridis"),
                dimensions=dimensions,
                visible=visibility,
            )
        )
    return data


def drop_identical(source_df: pd.DataFrame) -> pd.DataFrame:
    """
  Drop columns with identical values throughout iterations (no variance).
  :param source_df: Result of the hyperparameter run as a Dataframe
  :return source_df: Our source_df without identical-values parameters.
  """

    for col in (source_df.filter(like="param.")).columns:
        if source_df[col].nunique() == 1:
            source_df = source_df.drop(columns=[col])

    return source_df


def split_dataframe(
    source_df: pd.DataFrame, hide_identical: bool, exclude: list
) -> pd.DataFrame:
    """
    Splits the original hyperparameter dataframe into a params dataframe and result dataframe.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param hide_identical: Ignores parameters that remain the same throughout iterations
    :param exclude: User-provided list of parameters to be excluded from the graph
    :returns param_df: Dataframe of parameters from the hyperparameter run.
    :returns output_df: Dataframe of outputs from the hyperparameter run.
    """

    # Drop unwanted columns
    source_df = drop_exclusions(source_df, exclude)

    # Drop identical columns
    if hide_identical:
        source_df = drop_identical(source_df)

    # Create param and output dataframes
    param_df = source_df.filter(like="param.")
    param_df.columns = param_df.columns.str.replace("param.", "")

    output_df = source_df.filter(like="output.")
    output_df.columns = output_df.columns.str.replace("output.", "")

    return param_df, output_df


def plot_parallel_coordinates(
    source_df: pd.DataFrame = None,
    param_df: pd.DataFrame = None,
    output_df: pd.DataFrame = None,
    hide_identical: bool = True,
    exclude: list = [],
    display_plot=True,
) -> str:
    """
    Plots the output of the hyperparameter run in a Parallel Coordinate format using the Plotly library.
    :param source_df: Result of the hyperparameter run as a Dataframe
    :param param_df: Dataframe of parameters from the hyperparameter run
    :param output_df: Dataframe of outputs from the hyperparameter run
    :param hide_identical: Ignores parameters that remain the same throughout iterations
    :param exclude: User-provided list of parameters to be excluded from the graph
    :param display_plot: Allows the user to display the plot within the notebook
    :returns plot_as_html: The Parallel Coordinate plot in HTML format.
    """

    if source_df and param_df is None and output_df is None:
        param_df, output_df = split_dataframe(source_df, hide_identical, exclude)

    elif param_df and output_df and source_df is None:
        source_df = pd.concat([param_df, output_df], axis=1, join="inner")

    layout = dict(
        showlegend=True,
        updatemenus=list(
            [dict(active=0, showactive=True, buttons=gen_dropdown_buttons(output_df))]
        ),
    )

    fig = go.Figure(data=gen_plot_data(source_df, param_df, output_df), layout=layout)

    # Creating an html rendering of the plot
    plot_as_html = plotly.offline.plot(fig)

    if display_plot:
        display(HTML(plot_as_html))

    return plot_as_html
