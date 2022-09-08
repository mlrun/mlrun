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
