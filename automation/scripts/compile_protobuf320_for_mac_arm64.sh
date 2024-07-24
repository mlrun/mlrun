#!/bin/bash
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

set -e

MLRUN_PYTHON_PACKAGE_INSTALLER=${1:-${MLRUN_PYTHON_PACKAGE_INSTALLER:-pip}}
TARGET_PROTOBUF_VERSION=3.20.3

install_command=""
if [[ "${MLRUN_PYTHON_PACKAGE_INSTALLER}" == "pip" ]]; then
  install_command=$(echo pip install protobuf=="${TARGET_PROTOBUF_VERSION}" --force-reinstall --force --no-cache --no-binary :all: --global-option=\"--cpp_implementation\")
elif [[ "${MLRUN_PYTHON_PACKAGE_INSTALLER}" == "uv" ]]; then
  install_command=$(echo uv pip install --force-reinstall --no-cache --no-binary :all:  protobuf=="${TARGET_PROTOBUF_VERSION}")
else
    echo "Unknown package installer ${MLRUN_PYTHON_PACKAGE_INSTALLER}"
    exit 1
fi


# Install and use protobuf 3.20
brew install protobuf@3.20
brew link --overwrite protobuf@3.20

# Compile protobuf on mac
#   used https://github.com/protocolbuffers/protobuf/issues/8820#issuecomment-961552604
# Download and unpack protobuf compiler source code
PROTOC_SRC_ARCHIVE="protobuf-cpp-${TARGET_PROTOBUF_VERSION}.tar.gz" && \
    curl -sSL "https://github.com/protocolbuffers/protobuf/releases/download/v${TARGET_PROTOBUF_VERSION}/${PROTOC_SRC_ARCHIVE}" | tar -C /tmp -xzf -

# Build protobuf compiler from source (this will take a while)
#   see: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md for more info
PROTOC_SRC_PATH="/tmp/protobuf-${TARGET_PROTOBUF_VERSION}" && \
   pushd "${PROTOC_SRC_PATH}" && \
   ./configure && \
   make -j8 && \
   make check && \
   sudo make install && \
   popd && \
   rm -rf "${PROTOC_SRC_PATH}"

# Export protobuf to install package correctly
export PATH="/opt/homebrew/opt/protobuf@3/bin:$PATH"

# Install protobuf package
INSTALL_PREFIX_PATH="/usr/local" && \
   CFLAGS="-I${INSTALL_PREFIX_PATH}/include" && \
   LDFLAGS="-L${INSTALL_PREFIX_PATH}/lib" && \
   echo $install_command && $install_command
