#!/usr/bin/env sh
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


print_free_space() {
  df --human-readable
}

# before cleanup
print_free_space

# clean unneeded os packages and misc
sudo apt-get remove --yes '^dotnet-.*' 'php.*' azure-cli google-cloud-sdk google-chrome-stable firefox powershell
sudo apt-get autoremove --yes
sudo apt clean

# cleanup unneeded share dirs ~30GB
sudo rm --recursive --force \
    /usr/local/lib/android \
    /usr/share/dotnet \
    /usr/share/miniconda \
    /usr/share/swift \
    /opt/ghc \
    /usr/local/share/boost \
    "$AGENT_TOOLSDIRECTORY"

# remove large packages
sudo apt-get remove -y '^ghc-8.*'
sudo apt-get remove -y '^dotnet-.*'
sudo apt-get remove -y '^llvm-.*'
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel
sudo apt-get autoremove -y
sudo apt-get clean

# clean unneeded docker images
docker system prune --all --force

# post cleanup
print_free_space
