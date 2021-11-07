import pandas as pd
import plotly
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype, is_string_dtype


def create_index_list(df, column):
    values = list(df[column])
    index_list = []
    for val in values:
        index_list.append(values.index(val))
    return index_list


def gen_dimensions(df, column):
    dimension = {}

    if is_numeric_dtype(df[column]):
        dimension["range"] = [min(df[column]), max(df[column])]
        dimension["values"] = df[column]

    elif is_string_dtype(df[column]):
        dimension["range"] = [0, len(df[column])]
        dimension["values"] = create_index_list(df, column)
        dimension["tickvals"] = create_index_list(df, column)
        dimension["ticktext"] = list(df[column])

    # Axis name
    dimension["label"] = column

    return dimension


def gen_bool_list(output_df, col):
    bool_list = []
    for i in output_df.columns:
        if col == i:
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list


def gen_plot_data(all_df, param_df, output_df):

    data = []
    for output in output_df.columns:

        # Creating Axes and Dimensions
        dimensions = [(gen_dimensions(param_df, col)) for col in param_df.columns]
        dimensions.insert(0, gen_dimensions(all_df, "iter"))
        dimensions.append(gen_dimensions(output_df, output))

        # Plot Visibility on Dropdown
        visibility = True if output == output_df.columns[0] else False

        # Appending to list
        data.append(
            go.Parcoords(
                line=dict(
                    color=all_df["iter"],
                    colorscale=[[0, "purple"], [0.5, "lightseagreen"], [1, "gold"]],
                ),
                dimensions=dimensions,
                visible=visibility,
            )
        )
    return data


def gen_dropdown_buttons(output_df):
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


def plot_parallel_coordinates(all_df, param_df, output_df):
    updatemenus = list(
        [dict(active=0, showactive=True, buttons=gen_dropdown_buttons(output_df))]
    )

    layout = dict(showlegend=True, updatemenus=updatemenus)

    fig = go.Figure(data=gen_plot_data(all_df, param_df, output_df), layout=layout)

    # py.iplot(fig)

    return plotly.offline.plot(fig)


def gen_dataframes(run, exclude=["param.label_column", "param.sample"]):

    # Create dataframe from GridSearch
    all_df = pd.DataFrame(
        run["status"]["iterations"][1:], columns=run["status"]["iterations"][0]
    )

    # Drop unwanted columns
    all_df = all_df.drop(columns=exclude)

    # Create param and output dataframes
    param_df = all_df.filter(like="param.")
    output_df = all_df.filter(like="output.")
    return all_df, param_df, output_df
