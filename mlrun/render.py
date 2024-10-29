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
import pathlib
import typing
import uuid
from os import environ, path

import pandas as pd

import mlrun.utils

from .config import config
from .datastore import uri_to_ipython
from .utils import dict_to_list, get_in, is_jupyter

JUPYTER_SERVER_ROOT = environ.get("HOME", "/User")
supported_viewers = [
    ".htm",
    ".html",
    ".json",
    ".yaml",
    ".txt",
    ".log",
    ".jpg",
    ".png",
    ".csv",
    ".py",
]


def html_dict(title, data, open=False, show_nil=False):
    if not data:
        return ""
    html = ""
    for key, val in data.items():
        if show_nil or val:
            html += f"<tr><th>{key}</th><td>{val}</td></tr>"
    if html:
        html = f"<table>{html}</table>"
        return html_summary(title, html, open=open)
    return ""


def html_summary(title, data, num=None, open=False):
    tag = ""
    if open:
        tag = " open"
    if num:
        title = f"{title} ({num})"
    return f"<details{tag}><summary><b>{title}<b></summary>{data}</details>"


def html_crop(x):
    return f'<div class="ellipsis" ondblclick="copyToClipboard(this)" title="{x} (dbl click to copy)">{x}</div>'


def table_sum(title, df):
    size = len(df.index)
    if size > 0:
        return html_summary(title, df.to_html(escape=False), size)


def dict_html(x):
    return "".join([f'<div class="dictlist">{i}</div>' for i in dict_to_list(x)])


def link_to_ipython(link: str):
    """
    Convert a link (e.g. v3io path) to a jupyter notebook local link.

    :param link: the link to convert
    :return:     the converted link and ref for expanding the file in the notebook
    """
    valid = pathlib.Path(link).suffix in supported_viewers
    ref = 'class="artifact" onclick="expandPanel(this)" paneName="result" '
    if "://" not in link:
        abs = path.abspath(link)
        if abs.startswith(JUPYTER_SERVER_ROOT) and valid:
            return abs.replace(JUPYTER_SERVER_ROOT, "/files"), ref
        else:
            return abs, ""
    else:
        newlink = uri_to_ipython(link)
        if newlink and valid:
            return "files/" + newlink, ref
    return link, ""


def link_html(text, link=""):
    if not link:
        link = text
    link, ref = link_to_ipython(link)
    return f'<div {ref}title="{link}">{text}</div>'


def artifacts_html(
    artifacts: list[dict],
    attribute_name: str = "path",
):
    """
    Generate HTML for a list of artifacts. The HTML will be a list of links to the artifacts to be presented in the
    jupyter notebook. The links will be clickable and will open the artifact in a new tab.

    :param artifacts:       contains a list of artifact dictionaries
    :param attribute_name:  the attribute of the artifact to use as the link text
    :return:                the generated HTML
    """
    if not artifacts:
        return ""
    html = ""

    for artifact in artifacts:
        attribute_value = artifact["spec"].get(attribute_name)
        key = artifact["metadata"]["key"]

        if not attribute_value:
            mlrun.utils.logger.warning(
                f"Artifact required attribute {attribute_name} is missing, omitting from output",
                artifact_key=key,
            )
            continue

        link, ref = link_to_ipython(attribute_value)
        html += f'<div {ref}title="{link}">{key}</div>'
    return html


def inputs_html(x):
    if not x:
        return ""
    html = ""
    for k, v in x.items():
        link, ref = link_to_ipython(v)
        html += f'<div {ref}title="{link}">{k}</div>'
    return html


def sources_list_html(x):
    if not x:
        return ""
    html = ""
    for src in x:
        v = src.get("path", "")
        link, ref = link_to_ipython(v)
        html += f'<div {ref}title="{link}">{src["name"]}</div>'
    return html


def run_to_html(results, display=True):
    html = html_dict("Metadata", results["metadata"])
    html += html_dict("Spec", results["spec"])
    html += html_dict("results", results["status"].get("results"), True, True)

    if "iterations" in results["status"]:
        iter = results["status"]["iterations"]
        if iter:
            df = pd.DataFrame(iter[1:], columns=iter[0]).set_index("iter")
            html += table_sum("Iterations", df)

    artifacts = results["status"].get("artifacts", None)
    if artifacts:
        df = pd.DataFrame(artifacts)
        if "description" not in df.columns.values:
            df["description"] = ""
        df = df[["key", "kind", "target_path", "description"]]
        df["target_path"] = df["target_path"].apply(link_html)
        html += table_sum("Artifacts", df)

    return ipython_display(html, display)


def ipython_display(html, display=True, alt_text=None):
    if display and html and is_jupyter:
        import IPython.display

        IPython.display.display(IPython.display.HTML(html))
    elif alt_text:
        print(alt_text)
    return html


def get_style():
    return f"""<style>
.dictlist {{
  background-color: {config.background_color};
  text-align: center;
  margin: 4px;
  border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}}
.artifact {{
  cursor: pointer;
  background-color: {config.background_color};
  text-align: left;
  margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;
}}
div.block.hidden {{
  display: none;
}}
.clickable {{
  cursor: pointer;
}}
.ellipsis {{
  display: inline-block;
  max-width: 60px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.master-wrapper {{
  display: flex;
  flex-flow: row nowrap;
  justify-content: flex-start;
  align-items: stretch;
}}
.master-tbl {{
  flex: 3
}}
.master-wrapper > div {{
  margin: 4px;
  padding: 10px;
}}
iframe.fileview {{
  border: 0 none;
  height: 100%;
  width: 100%;
  white-space: pre-wrap;
}}
.pane-header-title {{
  width: 80%;
  font-weight: 500;
}}
.pane-header {{
  line-height: 1;
  background-color: {config.background_color};
  padding: 3px;
}}
.pane-header .close {{
  font-size: 20px;
  font-weight: 700;
  float: right;
  margin-top: -5px;
}}
.master-wrapper .right-pane {{
  border: 1px inset silver;
  width: 40%;
  min-height: 300px;
  flex: 3
  min-width: 500px;
}}
.master-wrapper * {{
  box-sizing: border-box;
}}
</style>"""


jscripts = r"""<script>
function copyToClipboard(fld) {
    if (document.queryCommandSupported && document.queryCommandSupported('copy')) {
        var textarea = document.createElement('textarea');
        textarea.textContent = fld.innerHTML;
        textarea.style.position = 'fixed';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            return document.execCommand('copy'); // Security exception may be thrown by some browsers.
        } catch (ex) {

        } finally {
            document.body.removeChild(textarea);
        }
    }
}
function expandPanel(el) {
  const panelName = "#" + el.getAttribute('paneName');

  // Get the base URL of the current notebook
  var baseUrl = window.location.origin;

  // Construct the full URL
  var fullUrl = new URL(el.title, baseUrl).href;

  document.querySelector(panelName + "-title").innerHTML = fullUrl
  iframe = document.querySelector(panelName + "-body");

  const tblcss = `<style> body { font-family: Arial, Helvetica, sans-serif;}
    #csv { margin-bottom: 15px; }
    #csv table { border-collapse: collapse;}
    #csv table td { padding: 4px 8px; border: 1px solid silver;} </style>`;

  function csvToHtmlTable(str) {
    return '<div id="csv"><table><tr><td>' +  str.replace(/[\n\r]+$/g, '').replace(/[\n\r]+/g, '</td></tr><tr><td>')
      .replace(/,/g, '</td><td>') + '</td></tr></table></div>';
  }

  function reqListener () {
    if (fullUrl.endsWith(".csv")) {
      iframe.setAttribute("srcdoc", tblcss + csvToHtmlTable(this.responseText));
    } else {
      iframe.setAttribute("srcdoc", this.responseText);
    }
    console.log(this.responseText);
  }

  const oReq = new XMLHttpRequest();
  oReq.addEventListener("load", reqListener);
  oReq.open("GET", fullUrl);
  oReq.send();


  //iframe.src = fullUrl;
  const resultPane = document.querySelector(panelName + "-pane");
  if (resultPane.classList.contains("hidden")) {
    resultPane.classList.remove("hidden");
  }
}
function closePanel(el) {
  const panelName = "#" + el.getAttribute('paneName')
  const resultPane = document.querySelector(panelName + "-pane");
  if (!resultPane.classList.contains("hidden")) {
    resultPane.classList.add("hidden");
  }
}

</script>"""

tblframe = """
<div class="master-wrapper">
  <div class="block master-tbl">{}</div>
  <div id="result-pane" class="right-pane block hidden">
    <div class="pane-header">
      <span id="result-title" class="pane-header-title">Title</span>
      <span onclick="closePanel(this)" paneName="result" class="close clickable">&times;</span>
    </div>
    <iframe class="fileview" id="result-body"></iframe>
  </div>
</div>
"""


def get_tblframe(df, display, classes=None):
    table_html = df.to_html(
        escape=False, index=False, notebook=display, classes=classes
    )
    if not display:
        return table_html

    table = tblframe.format(table_html)
    rnd = "result" + str(uuid.uuid4())[:8]
    html = get_style() + jscripts + table.replace('="result', '="' + rnd)
    return ipython_display(html, display)


uid_template = '<div title="{}"><a href="{}/{}/{}/jobs/monitor/{}/overview" target="_blank" >...{}</a></div>'


def runs_to_html(
    df: pd.DataFrame,
    display: bool = True,
    classes: typing.Optional[typing.Union[str, list, tuple]] = None,
    short: bool = False,
):
    def time_str(x):
        try:
            return x.strftime("%b %d %H:%M:%S")
        except ValueError:
            return ""

    df["results"] = df["results"].apply(dict_html)
    df["start"] = df["start"].apply(time_str)
    df["parameters"] = df["parameters"].apply(dict_html)
    if config.resolve_ui_url():
        df["uid"] = df.apply(
            lambda x: uid_template.format(
                x.uid,
                config.resolve_ui_url(),
                config.ui.projects_prefix,
                x.project,
                x.uid,
                x.uid[-8:],
            ),
            axis=1,
        )
    else:
        df["uid"] = df["uid"].apply(lambda x: f'<div title="{x}">...{x[-6:]}</div>')

    if short:
        df.drop("project", axis=1, inplace=True)
        if df["iter"].nunique() == 1:
            df.drop("iter", axis=1, inplace=True)
        df.drop("labels", axis=1, inplace=True)
        df.drop("inputs", axis=1, inplace=True)
        df.drop("artifacts", axis=1, inplace=True)
        df.drop("artifact_uris", axis=1, inplace=True)
    else:
        df["labels"] = df["labels"].apply(dict_html)
        df["inputs"] = df["inputs"].apply(inputs_html)
        if df["artifacts"][0]:
            df["artifacts"] = df["artifacts"].apply(
                lambda artifacts: artifacts_html(artifacts, "target_path"),
            )
            df.drop("artifact_uris", axis=1, inplace=True)
        elif df["artifact_uris"][0]:
            df["artifact_uris"] = df["artifact_uris"].apply(dict_html)
            df.drop("artifacts", axis=1, inplace=True)
        else:
            df.drop("artifacts", axis=1, inplace=True)
            df.drop("artifact_uris", axis=1, inplace=True)

    def expand_error(x):
        if x["state"] == "error":
            title = str(x["error"])
            state = f'<div style="color: red;" title="{title}">{x["state"]}</div>'
            x["state"] = state
        return x

    df = df.apply(expand_error, axis=1)
    df.drop("error", axis=1, inplace=True)
    with pd.option_context("display.max_colwidth", None):
        return get_tblframe(df, display, classes=classes)


def artifacts_to_html(df, display=True, classes=None):
    def prod_htm(x):
        if not x or not isinstance(x, dict):
            return ""
        kind = get_in(x, "kind", "")
        uri = get_in(x, "uri", "")
        name = get_in(x, "name", "unknown")
        title = f"{kind}/{uri}"
        if "owner" in x:
            title += f" by {x['owner']}"
        return f'<div title="{title}" class="producer">{name}</div>'

    if "tree" in df.columns.values:
        df["tree"] = df["tree"].apply(html_crop)
    df["path"] = df["path"].apply(link_html)
    df["hash"] = df["hash"].apply(html_crop)
    df["sources"] = df["sources"].apply(sources_list_html)
    df["labels"] = df["labels"].apply(dict_html)
    df["producer"] = df["producer"].apply(prod_htm)
    df["updated"] = df["updated"].apply(lambda x: x.strftime("%b %d %H:%M:%S"))
    with pd.option_context("display.max_colwidth", None):
        return get_tblframe(df, display, classes=classes)
