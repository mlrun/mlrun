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
import pandas as pd
from .utils import is_ipython, get_in, dict_to_list


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


def link_html(text, link=''):
    if not link:
        link = text
    if not text.startswith('/'):
        return '<a href="{}">{}</a>'.format(text, text)


def dict_html(x):
    return ''.join([f'<div class="dictlist">{i}</div>'
                    for i in dict_to_list(x)])


def artifacts_html(x, pathcol='path'):
    if not x:
        return ''
    template = '<div class="artifact" title="{}"><a href="{}">{}</a></div>'
    html = [template.format(i[pathcol], i['key'], i[pathcol]) for i in x]
    return ''.join(html)


def run_to_html(results, display=True):
    html = html_dict('Metadata', results['metadata'])
    html += html_dict('Spec', results['spec'])
    html += html_dict('Outputs', results['status'].get('outputs'), True, True)

    if 'iterations' in results['status']:
        iter = results['status']['iterations']
        if iter:
            df = pd.DataFrame(iter[1:], columns=iter[0]).set_index('iter')
            html += table_sum('Iterations', df)

    artifacts = results['status'].get('output_artifacts', None)
    if artifacts:
        df = pd.DataFrame(artifacts)
        if 'description' not in df.columns.values:
            df['description'] = ''
        df = df[['key', 'kind', 'target_path', 'description']]
        df['target_path'] = df['target_path'].apply(link_html)
        html += table_sum('Artifacts', df)

    return ipython_display(html, display)


def ipython_display(html, display=True):
    if display and html and is_ipython:
        import IPython
        IPython.display.display(IPython.display.HTML(html))
    return html


style = """<style> 
  .dictlist {background-color: #b3edff; text-align: center; margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}
  .artifact {background-color: #ffe6cc; text-align: left; margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}
</style>"""


def runs_to_html(df, display=True):
    df['inputs'] = df['inputs'].apply(artifacts_html)
    df['artifacts'] = df['artifacts'].apply(lambda x: artifacts_html(x, 'target_path'))
    df['labels'] = df['labels'].apply(dict_html)
    df['parameters'] = df['parameters'].apply(dict_html)
    df['results'] = df['results'].apply(dict_html)
    df['start'] = df['start'].apply(lambda x: x.strftime("%b %d %H:%M:%S"))
    df['uid'] = df['uid'].apply(lambda x: '<div title="{}">...{}</div>'.format(x, x[-6:]))
    pd.set_option('display.max_colwidth', -1)

    html = style + df.to_html(escape=False, index=False, notebook=True)
    return ipython_display(html, display)


def artifacts_to_html(df, display=True):
    def prod_htm(x):
        if not x or not isinstance(x, dict):
            return ''
        p = '{}/{}'.format(get_in(x, 'kind', ''), get_in(x, 'uri', ''))
        if 'owner' in x:
            p += ' by {}'.format(x['owner'])
        return '<div title="{}" class="producer">{}</div>'.format(p, get_in(x, 'name', 'unknown'))

    df['path'] = df['path'].apply(lambda x: f'<a href="{x}">{x}</a>')
    df['hash'] = df['hash'].apply(lambda x: '<div title="{}">...{}</div>'.format(x, x[-6:]))
    df['sources'] = df['sources'].apply(artifacts_html)
    df['labels'] = df['labels'].apply(dict_html)
    df['producer'] = df['producer'].apply(prod_htm)
    df['updated'] = df['updated'].apply(lambda x: x.strftime("%b %d %H:%M:%S"))
    pd.set_option('display.max_colwidth', -1)

    html = style + df.to_html(escape=False, index=False, notebook=True)
    return ipython_display(html, display)
