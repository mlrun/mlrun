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

from urllib.parse import urlparse

import s3fs


class S3FileSystemWithDS(s3fs.S3FileSystem):
    @classmethod
    def _strip_protocol(cls, url):
        if url.startswith("ds://"):
            parsed_url = urlparse(url)
            url = parsed_url.path[1:]
        return super()._strip_protocol(url)
