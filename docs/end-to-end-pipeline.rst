Demonstrate Local Or Remote Functions And Full Pipelines
========================================================

--------------

Creating a local function, running predefined functions, creating and
running a full ML pipeline with local and library functions.

**notebook how-to’s**
^^^^^^^^^^^^^^^^^^^^^

-  Create and test a simple function
-  Examine data using serverless (containarized) ``describe`` function
-  Create an automated ML pipeline from various library functions
-  Running and tracking the pipeline results and artifacts

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/lKEROW0GJHQ" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


Create and Test a Local Function (Iris Data Generator)
------------------------------------------------------

Import nuclio SDK and magics, do not remove the cell and comment !!!

.. code:: ipython3

    # nuclio: ignore
    import nuclio

Specify function dependencies and configuration

.. code:: ipython3

    %%nuclio cmd -c
    pip install sklearn
    pip install pyarrow

.. code:: ipython3

    %nuclio config spec.build.baseImage = "mlrun/mlrun"


.. parsed-literal::

    %nuclio: setting spec.build.baseImage to 'mlrun/mlrun'


Function code
^^^^^^^^^^^^^

Generate the iris dataset and log the dataframe (as csv or parquet file)

.. code:: ipython3

    import os
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import accuracy_score
    from mlrun.artifacts import TableArtifact, PlotArtifact
    import pandas as pd
    
    def iris_generator(context, format='csv'):
        iris = load_iris()
        iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_labels = pd.DataFrame(data=iris.target, columns=['label'])
        iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)
        
        context.logger.info('saving iris dataframe to {}'.format(context.artifact_path))
        context.log_dataset('iris_dataset', df=iris_dataset, format=format, index=False)


The following end-code annotation tells ``nuclio`` to stop parsing the
notebook from this cell. **Please do not remove this cell**:

.. code:: ipython3

    # nuclio: end-code
    # marks the end of a code section

Create a project to host our functions, jobs and artifacts
----------------------------------------------------------

Projects are used to package multiple functions, workflows, and
artifacts. We usually store project code and definitions in a Git
archive.

The following code creates a new project in a local dir and initialize
git tracking on that

.. code:: ipython3

    from os import path
    from mlrun import run_local, NewTask, mlconf, import_function, mount_v3io
    mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'
    
    # specify artifacts target location
    artifact_path = mlconf.artifact_path or path.abspath('./')
    project_name = 'sk-project'

.. code:: ipython3

    from mlrun import new_project, code_to_function
    project_dir = './project'
    skproj = new_project(project_name, project_dir, init_git=True)

 ### Run the data generator function locally

The functions above can be tested locally. Parameters, inputs, and
outputs can be specified in the API or the ``Task`` object. when using
``run_local()`` the function inputs and outputs are automatically
recorded by MLRun experiment and data tracking DB.

In each run we can specify the function, inputs,
parameters/hyper-parameters, etc… For more details, see the
`mlrun_basics notebook <mlrun_basics.ipynb>`__.

.. code:: ipython3

    # run the function locally
    gen = run_local(name='iris_gen', handler=iris_generator, 
                    project=project_name, artifact_path=path.join(artifact_path, 'data')) 


.. parsed-literal::

    [mlrun] 2020-05-20 11:54:56,925 starting run iris_gen uid=95d9058eac2d48bdb54352e78ff57bcd  -> http://mlrun-api:8080
    [mlrun] 2020-05-20 11:54:57,188 saving iris dataframe to /User/artifacts/data
    [mlrun] 2020-05-20 11:54:57,268 log artifact iris_dataset at /User/artifacts/data/iris_dataset.csv, size: 2776, db: Y
    



.. raw:: html

    <style> 
    .dictlist {
      background-color: #b3edff; 
      text-align: center; 
      margin: 4px; 
      border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}
    .artifact {
      cursor: pointer; 
      background-color: #ffe6cc; 
      text-align: left; 
      margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;
    }
    div.block.hidden {
      display: none;
    }
    .clickable {
      cursor: pointer;
    }
    .ellipsis {
      display: inline-block;
      max-width: 60px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .master-wrapper {
      display: flex;
      flex-flow: row nowrap;
      justify-content: flex-start;
      align-items: stretch;
    }
    .master-tbl {
      flex: 3
    }
    .master-wrapper > div {
      margin: 4px;
      padding: 10px;
    }
    iframe.fileview {
      border: 0 none;
      height: 100%;
      width: 100%;
      white-space: pre-wrap;
    }
    .pane-header-title {
      width: 80%;
      font-weight: 500;
    }
    .pane-header {
      line-height: 1;
      background-color: #ffe6cc;
      padding: 3px;
    }
    .pane-header .close {
      font-size: 20px;
      font-weight: 700;
      float: right;
      margin-top: -5px;
    }
    .master-wrapper .right-pane {
      border: 1px inset silver;
      width: 40%;
      min-height: 300px;
      flex: 3
      min-width: 500px;
    }
    .master-wrapper * {
      box-sizing: border-box;
    }
    </style><script>
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
    
    </script>
    <div class="master-wrapper">
      <div class="block master-tbl"><div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>project</th>
          <th>uid</th>
          <th>iter</th>
          <th>start</th>
          <th>state</th>
          <th>name</th>
          <th>labels</th>
          <th>inputs</th>
          <th>parameters</th>
          <th>results</th>
          <th>artifacts</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>sk-project</td>
          <td><div title="95d9058eac2d48bdb54352e78ff57bcd"><a href="https://mlrun-ui.default-tenant.app.yjb-mlrun-hope.iguazio-cd1.com/projects/sk-project/jobs/95d9058eac2d48bdb54352e78ff57bcd/info" target="_blank" >...8ff57bcd</a></div></td>
          <td>0</td>
          <td>May 20 11:54:56</td>
          <td>completed</td>
          <td>iris_gen</td>
          <td><div class="dictlist">v3io_user=admin</div><div class="dictlist">kind=handler</div><div class="dictlist">owner=admin</div><div class="dictlist">host=jupyter-67c88b95d4-crdhq</div></td>
          <td></td>
          <td></td>
          <td></td>
          <td><div class="artifact" onclick="expandPanel(this)" paneName="result022d42b0" title="/files/artifacts/data/iris_dataset.csv">iris_dataset</div></td>
        </tr>
      </tbody>
    </table>
    </div></div>
      <div id="result022d42b0-pane" class="right-pane block hidden">
        <div class="pane-header">
          <span id="result022d42b0-title" class="pane-header-title">Title</span>
          <span onclick="closePanel(this)" paneName="result022d42b0" class="close clickable">&times;</span>
        </div>
        <iframe class="fileview" id="result022d42b0-body"></iframe>
      </div>
    </div>



.. parsed-literal::

    to track results use .show() or .logs() or in CLI: 
    !mlrun get run 95d9058eac2d48bdb54352e78ff57bcd --project sk-project , !mlrun logs 95d9058eac2d48bdb54352e78ff57bcd --project sk-project
    [mlrun] 2020-05-20 11:54:57,373 run executed, status=completed


Convert our local code to a distributed serverless function object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    gen_func = code_to_function(name='gen_iris', kind='job')
    skproj.set_function(gen_func)




.. parsed-literal::

    <mlrun.runtimes.kubejob.KubejobRuntime at 0x7fc34f6f61d0>



Load and run a library function (visualize dataset features and stats)
----------------------------------------------------------------------

Step 1: load the function object from the function hub (marketplace) >
note: the function marketplace location is configurable, by default it
points to ``mlrun/functions`` git

.. code:: ipython3

    skproj.set_function('hub://describe', 'describe')




.. parsed-literal::

    <mlrun.runtimes.kubejob.KubejobRuntime at 0x7fc2f3956ac8>



.. code:: ipython3

    # read the remote function doc, params, usage
    skproj.func('describe').doc()
    #skproj.func('describe').spec.image_pull_policy = 'Always'


.. parsed-literal::

    function: describe
    describe and visualizes dataset stats
    default handler: summarize
    entry points:
      summarize: Summarize a table
        context(MLClientCtx)  - the function context
        table(DataItem)  - MLRun input pointing to pandas dataframe (csv/parquet file path)
        label_column(str)  - ground truth column label, default=labels
        class_labels(List[str])  - label for each class in tables and plots
        plot_hist(bool)  - (True) set this to False for large tables, default=True
        plots_dest(str)  - destination folder of summary plots (relative to artifact_path), default=plots


Step 2: Run the describe function as a Kubernetes job with specified
parameters.

   ``mount_v3io()`` vonnect our function to v3io shared file system and
   allow us to pass the data and get back the results (plots) directly
   to our notebook, we can choose other mount options to use NFS or
   object storage

.. code:: ipython3

    skproj.func('describe').apply(mount_v3io()).run(params={'label_column': 'label'}, 
                                                    inputs={"table": gen.outputs['iris_dataset']}, 
                                                    artifact_path=artifact_path)


.. parsed-literal::

    [mlrun] 2020-05-20 11:55:01,994 starting run describe-summarize uid=9fc84dd77c4142af995c33244ef870b6  -> http://mlrun-api:8080
    [mlrun] 2020-05-20 11:55:02,173 Job is running in the background, pod: describe-summarize-x6r9q
    [mlrun] 2020-05-20 11:55:12,627 starting local run: main.py # summarize
    [mlrun] 2020-05-20 11:55:16,068 log artifact histograms at /User/artifacts/plots/hist.html, size: 282853, db: Y
    [mlrun] 2020-05-20 11:55:16,597 log artifact imbalance at /User/artifacts/plots/imbalance.html, size: 11716, db: Y
    [mlrun] 2020-05-20 11:55:16,765 log artifact correlation at /User/artifacts/plots/corr.html, size: 30642, db: Y
    
    [mlrun] 2020-05-20 11:55:16,837 run executed, status=completed
    final state: succeeded



.. raw:: html

    <style> 
    .dictlist {
      background-color: #b3edff; 
      text-align: center; 
      margin: 4px; 
      border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}
    .artifact {
      cursor: pointer; 
      background-color: #ffe6cc; 
      text-align: left; 
      margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;
    }
    div.block.hidden {
      display: none;
    }
    .clickable {
      cursor: pointer;
    }
    .ellipsis {
      display: inline-block;
      max-width: 60px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .master-wrapper {
      display: flex;
      flex-flow: row nowrap;
      justify-content: flex-start;
      align-items: stretch;
    }
    .master-tbl {
      flex: 3
    }
    .master-wrapper > div {
      margin: 4px;
      padding: 10px;
    }
    iframe.fileview {
      border: 0 none;
      height: 100%;
      width: 100%;
      white-space: pre-wrap;
    }
    .pane-header-title {
      width: 80%;
      font-weight: 500;
    }
    .pane-header {
      line-height: 1;
      background-color: #ffe6cc;
      padding: 3px;
    }
    .pane-header .close {
      font-size: 20px;
      font-weight: 700;
      float: right;
      margin-top: -5px;
    }
    .master-wrapper .right-pane {
      border: 1px inset silver;
      width: 40%;
      min-height: 300px;
      flex: 3
      min-width: 500px;
    }
    .master-wrapper * {
      box-sizing: border-box;
    }
    </style><script>
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
    
    </script>
    <div class="master-wrapper">
      <div class="block master-tbl"><div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>project</th>
          <th>uid</th>
          <th>iter</th>
          <th>start</th>
          <th>state</th>
          <th>name</th>
          <th>labels</th>
          <th>inputs</th>
          <th>parameters</th>
          <th>results</th>
          <th>artifacts</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>sk-project</td>
          <td><div title="9fc84dd77c4142af995c33244ef870b6"><a href="https://mlrun-ui.default-tenant.app.yjb-mlrun-hope.iguazio-cd1.com/projects/sk-project/jobs/9fc84dd77c4142af995c33244ef870b6/info" target="_blank" >...4ef870b6</a></div></td>
          <td>0</td>
          <td>May 20 11:55:13</td>
          <td>completed</td>
          <td>describe-summarize</td>
          <td><div class="dictlist">host=describe-summarize-x6r9q</div><div class="dictlist">kind=job</div><div class="dictlist">owner=admin</div><div class="dictlist">v3io_user=admin</div></td>
          <td><div title="store://sk-project/iris_gen_iris_dataset#95d9058eac2d48bdb54352e78ff57bcd">table</div></td>
          <td><div class="dictlist">label_column=label</div></td>
          <td><div class="dictlist">scale_pos_weight=1.00</div></td>
          <td><div class="artifact" onclick="expandPanel(this)" paneName="resultd0de5a37" title="/files/artifacts/plots/hist.html">histograms</div><div class="artifact" onclick="expandPanel(this)" paneName="resultd0de5a37" title="/files/artifacts/plots/imbalance.html">imbalance</div><div class="artifact" onclick="expandPanel(this)" paneName="resultd0de5a37" title="/files/artifacts/plots/corr.html">correlation</div></td>
        </tr>
      </tbody>
    </table>
    </div></div>
      <div id="resultd0de5a37-pane" class="right-pane block hidden">
        <div class="pane-header">
          <span id="resultd0de5a37-title" class="pane-header-title">Title</span>
          <span onclick="closePanel(this)" paneName="resultd0de5a37" class="close clickable">&times;</span>
        </div>
        <iframe class="fileview" id="resultd0de5a37-body"></iframe>
      </div>
    </div>



.. parsed-literal::

    to track results use .show() or .logs() or in CLI: 
    !mlrun get run 9fc84dd77c4142af995c33244ef870b6 --project sk-project , !mlrun logs 9fc84dd77c4142af995c33244ef870b6 --project sk-project
    [mlrun] 2020-05-20 11:55:21,550 run executed, status=completed




.. parsed-literal::

    <mlrun.model.RunObject at 0x7fc2f7707438>



Create a Fully Automated ML Pipeline
------------------------------------

Add more functions to our project to be used in our pipeline (from the functions hub/marketplace)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AutoML training (classifier), Model validation (test_classifier),
Real-time model server, and Model REST API Tester

.. code:: ipython3

    skproj.set_function('hub://sklearn_classifier', 'train')
    skproj.set_function('hub://test_classifier', 'test')
    skproj.set_function('hub://model_server', 'serving')
    skproj.set_function('hub://model_server_tester', 'live_tester')
    #print(skproj.to_yaml())




.. parsed-literal::

    <mlrun.runtimes.kubejob.KubejobRuntime at 0x7fc2f38bc358>



Define and save a pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following workflow definition will be written into a file, it
describes a Kubeflow execution graph (DAG) and how functions and data
are connected to form an end to end pipeline.

-  Build the iris generator (ingest) function container
-  Ingest the iris data
-  Analyze the dataset (describe)
-  Train and test the model
-  Deploy the model as a real-time serverless function
-  Test the serverless function REST API with test dataset

Check the code below to see how functions objects are initialized and
used (by name) inside the workflow. The ``workflow.py`` file has two
parts, initialize the function objects and define pipeline dsl (connect
the function inputs and outputs).

   Note: the pipeline can include CI steps like building container
   images and deploying models as illustrated in the following example.

.. code:: ipython3

    %%writefile project/workflow.py
    from kfp import dsl
    from mlrun import mount_v3io
    
    funcs = {}
    DATASET = 'iris_dataset'
    LABELS  = "label"
    
    
    # init functions is used to configure function resources and local settings
    def init_functions(functions: dict, project=None, secrets=None):
        for f in functions.values():
            f.apply(mount_v3io())
         
        # uncomment this line to collect the inference results into a stream
        # and specify a path in V3IO (<datacontainer>/<subpath>)
        #functions['serving'].set_env('INFERENCE_STREAM', 'users/admin/model_stream')
    
        
    @dsl.pipeline(
        name="Demo training pipeline",
        description="Shows how to use mlrun."
    )
    def kfpipeline():
        
        # build our ingestion function (container image)
        builder = funcs['gen-iris'].deploy_step(skip_deployed=True)
        
        # run the ingestion function with the new image and params
        ingest = funcs['gen-iris'].as_step(
            name="get-data",
            handler='iris_generator',
            image=builder.outputs['image'],
            params={'format': 'pq'},
            outputs=[DATASET])
    
        # analyze our dataset
        describe = funcs["describe"].as_step(
            name="summary",
            params={"label_column": LABELS},
            inputs={"table": ingest.outputs[DATASET]})
        
        # train with hyper-paremeters 
        train = funcs["train"].as_step(
            name="train-skrf",
            params={"sample"          : -1, 
                    "label_column"    : LABELS,
                    "test_size"       : 0.10},
            hyperparams={'model_pkg_class': ["sklearn.ensemble.RandomForestClassifier", 
                                             "sklearn.linear_model.LogisticRegression",
                                             "sklearn.ensemble.AdaBoostClassifier"]},
            selector='max.accuracy',
            inputs={"dataset"         : ingest.outputs[DATASET]},
            outputs=['model', 'test_set'])
    
        # test and visualize our model
        test = funcs["test"].as_step(
            name="test",
            params={"label_column": LABELS},
            inputs={"models_path" : train.outputs['model'],
                    "test_set"    : train.outputs['test_set']})
    
        # deploy our model as a serverless function
        deploy = funcs["serving"].deploy_step(models={f"{DATASET}_v1": train.outputs['model']}, tag='v2')
        
        # test out new model server (via REST API calls)
        tester = funcs["live_tester"].as_step(name='model-tester',
            params={'addr': deploy.outputs['endpoint'], 'model': f"{DATASET}_v1"},
            inputs={'table': train.outputs['test_set']})



.. parsed-literal::

    Overwriting project/workflow.py


.. code:: ipython3

    # register the workflow file as "main", embed the workflow code into the project YAML
    skproj.set_workflow('main', 'workflow.py', embed=True)

Save the project definitions to a file (project.yaml), it is recommended
to commit all changes to a Git repo.

.. code:: ipython3

    skproj.save()

 ## Run a pipeline workflow use the ``run`` method to execute a
workflow, you can provide alternative arguments and specify the default
target for workflow artifacts. The workflow ID is returned and can be
used to track the progress or you can use the hyperlinks

   Note: The same command can be issued through CLI commands:
   ``mlrun project my-proj/ -r main -p "v3io:///users/admin/mlrun/kfp/{{workflow.uid}}/"``

The dirty flag allow us to run a project with uncommited changes (when
the notebook is in the same git dir it will always be dirty)

.. code:: ipython3

    artifact_path = path.abspath('./pipe/{{workflow.uid}}')
    run_id = skproj.run(
        'main',
        arguments={}, 
        artifact_path=artifact_path, 
        dirty=True)



.. raw:: html

    Experiment link <a href="https://dashboard.default-tenant.app.yjb-mlrun-hope.iguazio-cd1.com/pipelines/#/experiments/details/0cf2e8d1-d553-4c77-afff-1f60ba115c37" target="_blank" >here</a>



.. raw:: html

    Run link <a href="https://dashboard.default-tenant.app.yjb-mlrun-hope.iguazio-cd1.com/pipelines/#/runs/details/64d6f1e7-a582-4180-bba6-52c4a860d46b" target="_blank" >here</a>


.. parsed-literal::

    [mlrun] 2020-05-20 11:55:22,685 Pipeline run id=64d6f1e7-a582-4180-bba6-52c4a860d46b, check UI or DB for progress


Track pipeline results
^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from mlrun import get_run_db
    db = get_run_db().connect()
    db.list_runs(project=skproj.name, labels=f'workflow={run_id}').show()



.. raw:: html

    <style> 
    .dictlist {
      background-color: #b3edff; 
      text-align: center; 
      margin: 4px; 
      border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;}
    .artifact {
      cursor: pointer; 
      background-color: #ffe6cc; 
      text-align: left; 
      margin: 4px; border-radius: 3px; padding: 0px 3px 1px 3px; display: inline-block;
    }
    div.block.hidden {
      display: none;
    }
    .clickable {
      cursor: pointer;
    }
    .ellipsis {
      display: inline-block;
      max-width: 60px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .master-wrapper {
      display: flex;
      flex-flow: row nowrap;
      justify-content: flex-start;
      align-items: stretch;
    }
    .master-tbl {
      flex: 3
    }
    .master-wrapper > div {
      margin: 4px;
      padding: 10px;
    }
    iframe.fileview {
      border: 0 none;
      height: 100%;
      width: 100%;
      white-space: pre-wrap;
    }
    .pane-header-title {
      width: 80%;
      font-weight: 500;
    }
    .pane-header {
      line-height: 1;
      background-color: #ffe6cc;
      padding: 3px;
    }
    .pane-header .close {
      font-size: 20px;
      font-weight: 700;
      float: right;
      margin-top: -5px;
    }
    .master-wrapper .right-pane {
      border: 1px inset silver;
      width: 40%;
      min-height: 300px;
      flex: 3
      min-width: 500px;
    }
    .master-wrapper * {
      box-sizing: border-box;
    }
    </style><script>
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
    
    </script>
    <div class="master-wrapper">
      <div class="block master-tbl"><div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>project</th>
          <th>uid</th>
          <th>iter</th>
          <th>start</th>
          <th>state</th>
          <th>name</th>
          <th>labels</th>
          <th>inputs</th>
          <th>parameters</th>
          <th>results</th>
          <th>artifacts</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>sk-project</td>
          <td><div title="1f4a3e85b6f14d2387c6c4f7671c5df2"><a href="https://mlrun-ui.default-tenant.app.yh48.iguazio-cd2.com/projects/sk-project/jobs/1f4a3e85b6f14d2387c6c4f7671c5df2/info" target="_blank" >...671c5df2</a></div></td>
          <td>0</td>
          <td>Apr 10 20:51:29</td>
          <td>running</td>
          <td>train-skrf</td>
          <td><div class="dictlist">kind=job</div><div class="dictlist">owner=admin</div><div class="dictlist">v3io_user=admin</div><div class="dictlist">workflow=cfce7566-0446-400c-bb88-8688d7776c91</div></td>
          <td><div title="/User/ml/demos/sklearn-pipe/pipe/cfce7566-0446-400c-bb88-8688d7776c91/iris_dataset.parquet">dataset</div></td>
          <td><div class="dictlist">label_column=label</div><div class="dictlist">sample=-1</div><div class="dictlist">test_size=0.1</div></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td>sk-project</td>
          <td><div title="037fbb4babe54eb284f22901ce1fa27f"><a href="https://mlrun-ui.default-tenant.app.yh48.iguazio-cd2.com/projects/sk-project/jobs/037fbb4babe54eb284f22901ce1fa27f/info" target="_blank" >...ce1fa27f</a></div></td>
          <td>0</td>
          <td>Apr 10 20:51:29</td>
          <td>running</td>
          <td>summary</td>
          <td><div class="dictlist">kind=job</div><div class="dictlist">owner=admin</div><div class="dictlist">v3io_user=admin</div><div class="dictlist">workflow=cfce7566-0446-400c-bb88-8688d7776c91</div></td>
          <td><div title="/User/ml/demos/sklearn-pipe/pipe/cfce7566-0446-400c-bb88-8688d7776c91/iris_dataset.parquet">table</div></td>
          <td><div class="dictlist">label_column=label</div></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td>sk-project</td>
          <td><div title="b013bb2a7ff34dd788bead224e574ffd"><a href="https://mlrun-ui.default-tenant.app.yh48.iguazio-cd2.com/projects/sk-project/jobs/b013bb2a7ff34dd788bead224e574ffd/info" target="_blank" >...4e574ffd</a></div></td>
          <td>0</td>
          <td>Apr 10 20:51:20</td>
          <td>completed</td>
          <td>get-data</td>
          <td><div class="dictlist">host=get-data-mkrmx</div><div class="dictlist">kind=job</div><div class="dictlist">owner=admin</div><div class="dictlist">v3io_user=admin</div><div class="dictlist">workflow=cfce7566-0446-400c-bb88-8688d7776c91</div></td>
          <td></td>
          <td><div class="dictlist">format=pq</div></td>
          <td></td>
          <td><div title="/User/ml/demos/sklearn-pipe/pipe/cfce7566-0446-400c-bb88-8688d7776c91/iris_dataset.parquet">iris_dataset</div></td>
        </tr>
      </tbody>
    </table>
    </div></div>
      <div id="result1dd81966-pane" class="right-pane block hidden">
        <div class="pane-header">
          <span id="result1dd81966-title" class="pane-header-title">Title</span>
          <span onclick="closePanel(this)" paneName="result1dd81966" class="close clickable">&times;</span>
        </div>
        <iframe class="fileview" id="result1dd81966-body"></iframe>
      </div>
    </div>



`back to top <#top>`__
