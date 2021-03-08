import pandas
import pytest

import mlrun.mlutils.data


def test_get_sample_failure_label_not_exist():
    data = {"col1": [1, 2], "col2": [3, 4]}
    data_frame = pandas.DataFrame(data=data)
    with pytest.raises(ValueError):
        mlrun.mlutils.data.get_sample(data_frame, 2, "non_existing_label")
    with pytest.raises(ValueError):
        mlrun.mlutils.data.get_sample(data_frame, -2, "non_existing_label")
