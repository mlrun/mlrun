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
demos_dir=$(echo "/v3io/users/${user}")
pip_mlrun=$(pip show mlrun | grep Version)

# Grepping mlrun version
if [ -z "${pip_mlrun}" ]; then
    error_exit "MLRun version not found. Aborting..."
fi

# Extracting mlrun version and adding "-" between version prefix and rc
mlrun_version=$(echo "${pip_mlrun##Version: }" | sed 's/\([0-9]\)rc/\1-rc/')
echo "Detected MLRun version: ${mlrun_version}"

# Verifying mlrun >= 1.7
tag_prefix=`echo ${mlrun_version} | cut -d . -f1-2`
if [[ "${tag_prefix}" < "1.7" ]]; then
    error_exit "MLRun version must be 1.7 or above, for older updates run: sh https://raw.githubusercontent.com/mlrun/demos/v1.6.0/update_demos.sh .Aborting..."
fi

# copy & remove old demos folder
dt=$(date '+%Y%m%d%H%M%S');
old_demos_dir="${HOME}/demos.old/${dt}"
demos_dir="${HOME}/demos"
echo "Moving existing '${demos_dir}' to ${old_demos_dir}'..."
mkdir -p "${old_demos_dir}"
cp -r "${demos_dir}/." "${old_demos_dir}" && rm -rf "${demos_dir}"

# Making of git url that holds tar asset
# e.g. https://github.com/mlrun/mlrun/releases/download/v1.7.0-rc3/mlrun-demos.tar 
tar_url=$(echo "https://github.com/mlrun/mlrun/releases/download/v${mlrun_version}/mlrun-demos.tar")

# Downloading & extracting tar 
wget ${tar_url} 
mkdir ${demos_dir}
tar -xvf mlrun-demos.tar -C ${demos_dir} --strip-components 1

# Cleaning
rm -rf demos.tar
