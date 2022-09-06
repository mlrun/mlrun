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
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from cloudpickle import dumps
from sklearn.preprocessing import LabelEncoder

from mlrun.datastore import DataItem
from mlrun.execution import MLClientCtx


def data_clean(
    context: MLClientCtx,
    src: DataItem,
    file_ext: str = "csv",
    models_dest: str = "models/encoders",
    cleaned_key: str = "cleaned-data",
    encoded_key: str = "encoded-data",
):
    """process a raw churn data file

    Data has 3 states here: `raw`, `cleaned` and `encoded`

    * `raw` kept by default, the pipeline begins with a raw data artifact
    * `cleaned` kept for charts, presentations
    * `encoded` is input for a cross validation and training function

    steps (not necessarily in correct order, some parallel)
    * column name maps
    * deal with nans and other types of missings/junk
    * label encode binary and ordinal category columns
    * create category ranges from numerical columns
    And finally,
    * test

    Why we don't one-hot-encode here? One hot encoding isn't a necessary
    step for all algorithms. It can also generate a very large feature
    matrix that doesn't need to be serialized (even if sparse).
    So we leave one-hot-encoding for the training step.

    What about scaling numerical columns? Same as why we don't one hot
    encode here. Do we scale before train-test split?  IMHO, no.  Scaling
    before splitting introduces a type of data leakage.  In addition,
    many estimators are completely immune to the monotonic transformations
    implied by scaling, so why waste the cycles?

    TODO:
        * parallelize where possible
        * more abstraction (more parameters, chain sklearn transformers)
        * convert to marketplace function

    :param context:          the function execution context
    :param src:              an artifact or file path
    :param file_ext:         file type for artifacts
    :param models_dest:       label encoders and other preprocessing steps
                             should be saved together with other pipeline
                             models
    :param cleaned_key:      key of cleaned data table in artifact store
    :param encoded_key:      key of encoded data table in artifact store
    """
    df = src.as_df()

    # drop columns
    drop_cols_list = ["customerID", "TotalCharges"]
    df.drop(drop_cols_list, axis=1, inplace=True)

    # header transformations
    rename_cols_map = {
        "SeniorCitizen": "senior",
        "Partner": "partner",
        "Dependents": "deps",
        "Churn": "labels",
    }
    df.rename(rename_cols_map, axis=1, inplace=True)

    # add drop column to logs:
    for col in drop_cols_list:
        rename_cols_map.update({col: "_DROPPED_"})

    # log the op
    tp = os.path.join(models_dest, "preproc-column_map.json")
    context.log_artifact(
        "preproc-column_map.json", body=json.dumps(rename_cols_map), local_path=tp
    )

    # VALUE transformations

    # clean
    # truncate reply to "No"
    df = df.applymap(lambda x: "No" if str(x).startswith("No ") else x)

    # encode numerical type as category bins (ordinal)
    bins = [0, 12, 24, 36, 48, 60, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    df["tenure_map"] = pd.cut(df.tenure, bins, labels=False)
    tenure_map = dict(zip(bins, labels))
    # save this transformation
    tp = os.path.join(models_dest, "preproc-numcat_map.json")
    context.log_artifact(
        "preproc-numcat_map.json",
        body=bytes(json.dumps(tenure_map).encode("utf-8")),
        local_path=tp,
    )

    context.log_dataset(cleaned_key, df=df, format=file_ext, index=False)

    # label encoding - generate model for each column saved in dict
    # some of these columns may be hot encoded in the training step
    fix_cols = [
        "gender",
        "partner",
        "deps",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PhoneService",
        "MultipleLines",
        "PaperlessBilling",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "labels",
    ]

    d = defaultdict(LabelEncoder)
    df[fix_cols] = df[fix_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    context.log_dataset(encoded_key, df=df, format=file_ext, index=False)

    model_bin = dumps(d)
    context.log_model(
        "model",
        body=model_bin,
        artifact_path=os.path.join(context.artifact_path, models_dest),
        model_file="model.pkl",
    )
