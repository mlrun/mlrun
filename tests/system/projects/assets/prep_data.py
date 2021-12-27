import mlrun


def prep_data(context, source_url: mlrun.DataItem, label_column="label"):
    # Convert the DataItem to a pandas DataFrame
    df = source_url.as_df()
    print("data url:", source_url.url)
    df[label_column] = df[label_column].astype("category").cat.codes

    # Record the DataFrane length after the run
    context.log_result("num_rows", df.shape[0])

    # Store the data set in your artifacts database
    context.log_dataset("cleaned_data", df=df, index=False, format="csv")
