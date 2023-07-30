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
import os
import zipfile


def convert_np_dtype_to_value_type():
    """
    needed to avoid import issues later
    :return: CommonUtils.convert_np_dtype_to_value_type
    """
    from mlrun.frameworks._common import CommonUtils

    convert_func = CommonUtils.convert_np_dtype_to_value_type
    return convert_func


def zip_folder(folder_path: str, output_path: str):
    """
    creates a zip copy of given dir
    :param folder_path: path of dir to zip
    :param output_path: designated path of dir after zip
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
