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
import base64
from io import BytesIO


def html_dict(title, data, open=False, show_nil=False):
    if not data:
        return ''
    html = ''
    for key, val in data.items():
        if show_nil or val:
            html += f'<tr><th>{key}</th><td>{val}</td></tr>'
    if html:
        html = f'<table>{html}</table>'
        return html_summary(title, html, open=open)
    return ''


def html_summary(title, data, num=None, open=False):
    tag = ''
    if open:
        tag = ' open'
    if num:
        title = f'{title} ({num})'
    summary = '<details{}><summary><b>{}<b></summary>{}</details>'
    return summary.format(tag, title, data)


def table_sum(title, df):
    size = len(df.index)
    if size > 0:
        return html_summary(title, df.to_html(escape=False), size)


def plot_to_html(fig):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    canvas = FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    data = png_output.getvalue()

    data_uri = base64.b64encode(data).decode('utf-8')
    return '<img src="data:image/png;base64,{0}">'.format(data_uri)


def ipython_ui(results, show=True):

    html = html_dict('Metadata', results['metadata'])
    html += html_dict('Spec', results['spec'])
    html += html_dict('Outputs', results['status'].get('outputs'), True, True)

    def to_url(text=''):
        if not text.startswith('/'):
            return '<a href="{}">{}</a>'.format(text, text)

    if 'iterations' in results['status']:
        iter = results['status']['iterations']
        if iter:
            df = pd.DataFrame(iter[1:], columns=iter[0]).set_index('iter')
            html += table_sum('Iterations', df)

    artifacts = results['status'].get('output_artifacts', None)
    if artifacts:
        df = pd.DataFrame(artifacts)[['key', 'kind', 'target_path', 'description']]
        df['target_path'] = df['target_path'].apply(to_url)
        html += table_sum('Artifacts', df)

    if html and show:
        import IPython
        IPython.display.display(IPython.display.HTML(html))

    return html

