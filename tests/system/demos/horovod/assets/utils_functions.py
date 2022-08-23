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
import zipfile

import pandas as pd

from mlrun import DataItem


# download the image archive
def open_archive(context, archive_url: DataItem, target_path, refresh=False):
    """Open a file/object archive into a target directory

    Currently supports zip and tar.gz

    :param context:      function execution context
    :param archive_url:  url of archive file
    :param target_path:  file system path to store extracted files
    :param key:          key of archive contents in artifact store
    """
    os.makedirs(target_path, exist_ok=True)

    # get the archive as a local file (download if needed)
    archive_url = archive_url.local()

    context.logger.info("Extracting zip")
    zip_ref = zipfile.ZipFile(archive_url, "r")
    zip_ref.extractall(target_path)
    zip_ref.close()

    context.logger.info(f"extracted archive to {target_path}")
    context.log_artifact("content", target_path=target_path)


# build categories
def categories_map_builder(
    context,
    source_dir,
    df_filename="file_categories_df.csv",
    map_filename="categories_map.json",
):
    """Read labeled images from a directory and create category map + df

    filename format: <category>.NN.jpg"""

    # create filenames list (jpg only)
    filenames = [file for file in os.listdir(source_dir) if file.endswith(".jpg")]
    categories = []

    # Create a pandas DataFrame for the full sample
    for filename in filenames:
        category = filename.split(".")[0]
        categories.append(category)

    df = pd.DataFrame({"filename": filenames, "category": categories})
    df["category"] = df["category"].astype("str")

    categories = df.category.unique()
    categories = {i: category for i, category in enumerate(categories)}
    with open(os.path.join(context.artifact_path, map_filename), "w") as f:
        f.write(json.dumps(categories))

    context.logger.info(categories)
    context.log_artifact("categories_map", local_path=map_filename)
    context.log_dataset("file_categories", df=df, local_path=df_filename)
