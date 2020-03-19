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

"""mlrun database HTTP server"""

from argparse import ArgumentParser

from mlrun.config import config

from mlrun.httpd import app
from mlrun.httpd import routes  # noqa - register routes


# Don't remove this function, it's an entry point in setup.py
def main():
    parser = ArgumentParser(description=__doc__)
    parser.parse_args()
    app.run(
        host='0.0.0.0',
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )


if __name__ == '__main__':
    main()
