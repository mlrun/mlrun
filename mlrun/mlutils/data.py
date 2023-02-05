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
from typing import Union

import pandas as pd
from deprecated import deprecated
from sklearn.model_selection import train_test_split

from ..datastore import DataItem

# TODO: remove mlutils in 1.5.0


@deprecated(
    version="1.3.0",
    reason="'mlrun.mlutils' will be removed in 1.5.0, use 'mlrun.framework' instead",
    category=FutureWarning,
)
def get_sample(
    src: Union[DataItem, pd.core.frame.DataFrame], sample: int, label: str, reader=None
):
    """generate data sample to be split (candidate for mlrun)

    Return features matrix and header (x), and labels (y)
    :param src:    data artifact
    :param sample: sample size from data source, use negative
                   integers to sample randomly, positive to
                   sample consecutively from the first row
    :param label:  label column title
    """
    if type(src) == pd.core.frame.DataFrame:
        table = src
    else:
        table = src.as_df()

    # get sample
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = table.dropna()
        labels = _get_label_from_raw(raw, label)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = table.dropna().sample(sample * -1)
        labels = _get_label_from_raw(raw, label)

    return raw, labels, raw.columns.values


def _get_label_from_raw(raw, label):
    """
    Just a stupid wrapper so that nice error will be raised when users give wrong label
    """
    if label not in raw:
        raise ValueError(f"Specified label could not be found: {label}")
    return raw.pop(label)


@deprecated(
    version="1.3.0",
    reason="'mlrun.mlutils' will be removed in 1.5.0, use 'mlrun.framework' instead",
    category=FutureWarning,
)
def get_splits(
    raw,
    labels,
    n_ways: int = 3,
    test_size: float = 0.15,
    valid_size: float = 0.30,
    label_names: list = ["labels"],
    random_state: int = 1,
):
    """generate train and test sets (candidate for mlrun)
    cross validation:
    1. cut out a test set
    2a. use the training set in a cross validation scheme, or
    2b. make another split to generate a validation set

    2 parts (n_ways=2): train and test set only
    3 parts (n_ways=3): train, validation and test set

    :param raw:            dataframe or numpy array of raw features
    :param labels:         dataframe or numpy array of raw labels
    :param n_ways:         (3) split data into 2 or 3 parts
    :param test_size:      proportion of raw data to set asid as test data
    :param valid_size:     proportion of remaining data to be set as validation
    :param label_names:         label names
    :param random_state:   (1) random number seed
    """
    x, xte, y, yte = train_test_split(
        raw, labels, test_size=test_size, random_state=random_state
    )
    if n_ways == 2:
        return (x, y), (xte, yte)
    elif n_ways == 3:
        xtr, xva, ytr, yva = train_test_split(
            x, y, train_size=1 - valid_size, random_state=random_state
        )
        return (xtr, ytr), (xva, yva), (xte, yte)
    else:
        raise Exception("n_ways must be in the range [2,3]")


@deprecated(
    version="1.3.0",
    reason="'mlrun.mlutils' will be removed in 1.5.0, use 'mlrun.framework' instead",
    category=FutureWarning,
)
def save_test_set(
    context,
    data: dict,
    header: list,
    label: str = "labels",
    file_ext: str = "parquet",
    index: bool = False,
    debug: bool = False,
):
    """log a held out test set
    :param context:    the function execution context
    :param data:       dict with keys 'xtest'. 'ytest', and optionally
                       'xcal', 'ycal' if n_ways=4 in `get_splits`
    :param header:     ([])features header if required
    :param label:      ("labels") name of label column
    :param file_ext:   format of test set file
    :param index:      preserve index column
    :param debug:      (False)
    """

    if all(x in data.keys() for x in ["xtest", "ytest"]):
        test_set = pd.concat(
            [
                pd.DataFrame(data=data["xtest"], columns=header),
                pd.DataFrame(data=data["ytest"].values, columns=[label]),
            ],
            axis=1,
        )
        context.log_dataset("test_set", df=test_set, format=file_ext, index=index)

    if all(x in data.keys() for x in ["xcal", "ycal"]):
        cal_set = pd.concat(
            [
                pd.DataFrame(data=data["xcal"], columns=header),
                pd.DataFrame(data=data["ycal"].values, columns=[label]),
            ],
            axis=1,
        )
        context.log_dataset("calibration_set", df=cal_set, format=file_ext, index=index)
