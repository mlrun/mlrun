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
import uuid
import pandas as pd
from os import path, environ
import pathlib
from .utils import is_ipython, get_in, dict_to_list
from .datastore import uri_to_ipython
from .config import config

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
    summary = "<details{}><summary><b>{}<b></summary>{}</details>"
    return summary.format(tag, title, data)


def html_crop(x):
    return f'<div class="ellipsis" ondblclick="copyToClipboard(this)" title="{x} (dbl click to copy)">{x}</div>'


def table_sum(title, df):
    size = len(df.index)
    if size > 0:
        return html_summary(title, df.to_html(escape=False), size)


def dict_html(x):
    return "".join([f'<div class="dictlist">{i}</div>' for i in dict_to_list(x)])


def link_to_ipython(link):
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
    return '<div {}title="{}">{}</div>'.format(ref, link, text)


def artifacts_html(x, pathcol="path"):
    if not x:
        return ""
    html = ""
    for i in x:
        link, ref = link_to_ipython(i[pathcol])
        html += '<div {}title="{}">{}</div>'.format(ref, link, i["key"])
    return html


def inputs_html(x):
    if not x:
        return ""
    html = ""
    for k, v in x.items():
        link, ref = link_to_ipython(v)
        html += '<div {}title="{}">{}</div>'.format(ref, link, k)
    return html


def sources_list_html(x):
    if not x:
        return ""
    html = ""
    for src in x:
        v = src.get("path", "")
        link, ref = link_to_ipython(v)
        html += '<div {}title="{}">{}</div>'.format(ref, link, src["name"])
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
    if display and html and is_ipython:
        import IPython

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
  console.log(el.title);

  document.querySelector(panelName + "-title").innerHTML = el.title
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
    if (el.title.endsWith(".csv")) {
      iframe.setAttribute("srcdoc", tblcss + csvToHtmlTable(this.responseText));
    } else {
      iframe.setAttribute("srcdoc", this.responseText);
    }
    console.log(this.responseText);
  }

  const oReq = new XMLHttpRequest();
  oReq.addEventListener("load", reqListener);
  oReq.open("GET", el.title);
  oReq.send();


  //iframe.src = el.title;
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


uid_template = '<div title="{}"><a href="{}/projects/{}/jobs/monitor/{}/info" target="_blank" >...{}</a></div>'


def runs_to_html(df, display=True, classes=None, short=False):
    def time_str(x):
        try:
            return x.strftime("%b %d %H:%M:%S")
        except ValueError:
            return ""

    df["artifacts"] = df["artifacts"].apply(lambda x: artifacts_html(x, "target_path"))
    df["results"] = df["results"].apply(dict_html)
    df["start"] = df["start"].apply(time_str)
    if config.ui_url:
        df["uid"] = df.apply(
            lambda x: uid_template.format(
                x.uid, config.ui_url, x.project, x.uid, x.uid[-8:]
            ),
            axis=1,
        )
    else:
        df["uid"] = df["uid"].apply(
            lambda x: '<div title="{}">...{}</div>'.format(x, x[-6:])
        )

    if short:
        df.drop("project", axis=1, inplace=True)
        df.drop("iter", axis=1, inplace=True)
        df.drop("labels", axis=1, inplace=True)
        df.drop("inputs", axis=1, inplace=True)
        df.drop("parameters", axis=1, inplace=True)
    else:
        df["labels"] = df["labels"].apply(dict_html)
        df["inputs"] = df["inputs"].apply(inputs_html)
        df["parameters"] = df["parameters"].apply(dict_html)

    def expand_error(x):
        if x["state"] == "error":
            x["state"] = '<div style="color: red;" title="{}">{}</div>'.format(
                (str(x["error"])).replace('"', "'"), x["state"]
            )
        return x

    df = df.apply(expand_error, axis=1)
    df.drop("error", axis=1, inplace=True)
    with pd.option_context("display.max_colwidth", None):
        return get_tblframe(df, display, classes=classes)


def artifacts_to_html(df, display=True, classes=None):
    def prod_htm(x):
        if not x or not isinstance(x, dict):
            return ""
        p = "{}/{}".format(get_in(x, "kind", ""), get_in(x, "uri", ""))
        if "owner" in x:
            p += " by {}".format(x["owner"])
        return '<div title="{}" class="producer">{}</div>'.format(
            p, get_in(x, "name", "unknown")
        )

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
