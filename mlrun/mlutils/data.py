import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..datastore import DataItem


def get_sample(src: DataItem, sample: int, label: str, reader=None):
    """generate data sample to be split (candidate for mlrun)
     
    Returns features matrix and header (x), and labels (y)
    :param src:    data artifact
    :param sample: sample size from data source, use negative 
                   integers to sample randomly, positive to
                   sample consecutively from the first row
    :param label:  label column title
    """
    table = src.as_df()

    # get sample
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = table.dropna()
        labels = raw.pop(label)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = table.dropna().sample(sample * -1)
        labels = raw.pop(label)

    return raw, labels, raw.columns.values


def get_splits(
    raw,
    labels,
    n_ways: int = 3,
    test_size: float = 0.15,
    valid_size: float = 0.30,
    label_names: list = ["labels"],
    random_state: int = 1
):
    """generate train and test sets (candidate for mlrun)
    cross validation:
    1. cut out a test set
    2a. use the training set in a cross validation scheme, or
    2b. make another split to generate a validation set
    
    2 parts (n_ways=2): train and test set only
    3 parts (n_ways=3): train, validation and test set
    4 parts (n_ways=4): n_ways=3 + a held-out probability calibration set
    
    :param raw:            dataframe or numpy array of raw features
    :param labels:         dataframe or numpy array of raw labels
    :param n_ways:         (3) split data into 2, 3, or 4 parts
    :param test_size:      proportion of raw data to set asid as test data
    :param valid_size:     proportion of remaining data to be set as validation
    :param label_names:         label names
    :param random_state:   (1) random number seed
    """
    if isinstance(raw, np.ndarray):
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        xy = np.concatenate([raw, labels], axis=1)
    else:
        if isinstance(labels, pd.Series):
            labels = pd.DataFrame(data=labels, columns=label_names)
        xy = pd.concat([raw, labels], axis=1)

    x, xte, y, yte = train_test_split(xy, labels, test_size=test_size,
                                      random_state=random_state)
    if n_ways == 2:
        return (x, y), (xte, yte), None, None
    elif n_ways == 3:
        xtr, xva, ytr, yva = train_test_split(x, y, train_size=valid_size,
                                              random_state=random_state)
        return (xtr, ytr), (xva, yva), (xte, yte), None
    elif n_ways == 4:
        xt, xva, yt, yva = train_test_split(x, y, train_size=valid_size,
                                            random_state=random_state)
        xtr, xcal, ytr, ycal = train_test_split(xt, yt, train_size=0.8,
                                                random_state=random_state)
        return (xtr, ytr), (xva, yva), (xte, yte), (xcal, ycal)
    else:
        raise Exception("n_ways must be in the range [2,4]")


def save_test_set(
    context,
    data: dict,
    header: list,
    label: str = "labels",
    file_ext: str = "parquet",
    index: bool = False,
    debug: bool = False
):
    """log a held out test set
    :param context:    the function execution context
    :param data:       dict with keys 'xtest'. 'ytest', and optionally 
                       'xcal', 'ycal' if n_ways=4 in `get_splits`
    :param ytest:      test labels, as np.ndarray output from `get_splits`
    :param header:     ([])features header if required
    :param label:      ("labels") name of label column
    :param file_ext:   format of test set file
    :param index:      preserve index column
    :param debug:      (False)
    """
    import pandas as pd

    if all(x in data.keys() for x in ["xtest", "ytest"]):
        test_set = pd.concat(
            [pd.DataFrame(data=data["xtest"], columns=header),
             pd.DataFrame(data=data["ytest"].values, columns=[label])],
            axis=1)
        context.log_dataset(
            "test_set",
            df=test_set,
            format=file_ext,
            index=index)

    if all(x in data.keys() for x in ["xcal", "ycal"]):
        cal_set = pd.concat(
            [pd.DataFrame(data=data["xcal"], columns=header),
             pd.DataFrame(data=data["ycal"].values, columns=[label])],
            axis=1)
        context.log_dataset(
            "calibration_set",
            df=cal_set,
            format=file_ext,
            index=index)
