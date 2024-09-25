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

TARGET_PROTOBUF_VERSION=3.20.3

# uncomment above and delete this line once `uv` would be able to accept
# cpp_implementation as a global option.
install_command=$(echo pip install protobuf=="${TARGET_PROTOBUF_VERSION}" --force-reinstall --force --no-cache --no-binary :all: --global-option=\"--cpp_implementation\")


# Install and use protobuf 3.20
brew install --force protobuf@3.20

# Ensure that the correct version of protobuf is linked
brew unlink protobuf@3 && brew link --force protobuf@3

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
   $install_command
