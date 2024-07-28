# Copyright 2024 Iguazio
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

import vizro.models as vm
import vizro.plotly.express as px
from vizro import Vizro

df = px.data.iris()

page = vm.Page(
    title="My first dashboard",
    components=[
        vm.Graph(
            id="scatter_chart",
            figure=px.scatter(df, x="sepal_length", y="petal_width", color="species"),
        ),
        vm.Graph(
            id="hist_chart", figure=px.histogram(df, x="sepal_width", color="species")
        ),
    ],
    controls=[
        vm.Filter(column="species", selector=vm.Dropdown(value=["ALL"])),
    ],
)

dashboard = vm.Dashboard(pages=[page])
app = Vizro().build(dashboard)
server = app.dash.server

if __name__ == "__main__":
    app.run()
