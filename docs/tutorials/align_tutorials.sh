# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
error_exit()
{

# ----------------------------------------------------------------
# Function for exit due to fatal program error
#   Accepts 1 argument:
#     string containing descriptive error message
# ----------------------------------------------------------------
  echo "${SCRIPT}: ${1:-"Unknown Error"}" 1>&2
  exit 1
}

user=${V3IO_USERNAME}
tutorial_dir=$(echo "/v3io/users/${user}")
pip_mlrun=$(pip show mlrun | grep Version)

# Grepping mlrun version
if [ -z "${pip_mlrun}" ]; then
    error_exit "MLRun version not found. Aborting..."
fi

# Extracting mlrun version and adding "-" between version prefix and rc
mlrun_version=$(echo "${pip_mlrun##Version: }" | sed 's/\([0-9]\)rc/\1-rc/')
echo "Detected MLRun version: ${mlrun_version}"

# Verifying mlrun >= 1.4.0
tag_prefix=`echo ${mlrun_version} | cut -d . -f1-2`
if [[ "${tag_prefix}" < "1.4" ]]; then
    error_exit "MLRun version must be 1.4.0 or above. Aborting..."
fi

# removing old tutorial folder
rm -rf "${tutorial_dir}/tutorial"

# Making of git url that holds tar asset
# e.g. https://github.com/mlrun/mlrun/releases/download/v1.5.0-rc3/mlrun-tutorials.tar 
tar_url=$(echo "https://github.com/mlrun/mlrun/releases/download/v${mlrun_version}/mlrun-tutorials.tar")

# Downloading & extracting tar 
wget ${tar_url} 
tar -xvf mlrun-tutorials.tar -C ${tutorial_dir} --strip-components 1

# Cleaning
rm -rf mlrun-tutorials.tar
