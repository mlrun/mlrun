# Copyright 2023 Iguazio
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
import tempfile


def create_file_artifact() -> str:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    file_content = b"This is the content of the file."
    file_name = "example.txt"

    # Create the full path to the file within the temporary directory
    file_path = os.path.join(temp_dir, file_name)

    # Write the content to the file
    with open(file_path, "wb") as file:
        file.write(file_content)

    print(
        f"Temporary directory '{temp_dir}' created successfully with file '{file_name}'."
    )
    return str(temp_dir)
