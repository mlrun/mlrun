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
r"""
MLRun comes with the following list of modules, out of the box. All of the packagers listed here
use the implementation of :ref:`DefaultPackager <mlrun.package.packagers.default\_packager.DefaultPackager>` and are
available by default at the start of each run.
"""
# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from .default_packager import DefaultPackager
from .numpy_packagers import NumPySupportedFormat
