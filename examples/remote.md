# Using MLRurn from a remote client

MLRun can be used from a local IDE and run jobs on a remote cluster, you first need to:
1. Install mlrun locally (use: pip unstall mlrun)
2. Have remote access to your MLRun servive (node port on the remote k8s cluster)
3. set environment variables with appropriate information:

as a minimum specify the remote address:

    MLRUN_DBPATH=http://<cluster-ip>:<port>
    
If you want to use iguazio data services you need to add their credentials:    

    V3IO_USERNAME=<user-name>
    V3IO_API=<api endpoint, e.g.: webapi.default-tenant.app.xxx.iguazio-cd2.com>
    V3IO_ACCESS_KEY=<access_key>


## CLI commands

the `build` command will build all the function dependencies (docker image)
from the function spec 

```
  --name TEXT            function name
  --project TEXT         project name
  --tag TEXT             function tag
  -i, --image TEXT       location/url of the source files dir/tar
  -s, --source TEXT      location/url of the source files dir/tar
  -b, --base-image TEXT  base docker image
  -c, --command TEXT     build commands, e.g. '-c pip install pandas'
  --secret-name TEXT     container registry secret name
  -a, --archive TEXT     destination archive for code (tar)
  --silent               do not show build logs
  --with-mlrun           add MLRun package
```

the archive option allow defining a remote location e.g. s3 or v3io object path that will store tar files with all the 
code dependencies we need, it can also be set using the `MLRUN_DEFAULT_ARCHIVE` env var
 
 
 the `run` executes a task using a local or remote function it can accept many options, see `mlrun run --help` for details.
 the key ones are:
 
```
  -p, --param key=val    parameter name and value tuples, e.g. -p x=37 -p y='text'
  -i, --inputs key=path  input artifact url 
  --in-path TEXT         default input path/url (prefix) for artifact
  --out-path TEXT        default output path/url (prefix) for artifact
  -s, --secrets TEXT     secrets file=<filename> or env=ENV_KEY1,..
  --name TEXT            optional run name 
  --project TEXT         project name/id
  -f, --func-url TEXT    path/url of function yaml or db://<project>/<name>[:tag]
  --task TEXT            path/url to task yaml
  --handler TEXT         invoke function handler inside the code file
```



## Running function from a git repo

in your IDE working directory place a YAML file describing the 
function, example:

```yaml
kind: job
metadata:
  name: remote-demo1
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image: .mlrun/func-default-remote-demo-ps-latest
  image_pull_policy: Always
  build:
    #commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/mlrun
```

save it as `myfunc.yaml`

in the command line run mlrun build command to build the container image:

    mlrun build myfunc.yaml
    
the function will be built according to the specified requirements, once the build is complete we can run the function

    mlrun run -f myfunc.yaml -w -p p1=3
    

You can also try the following example based on the mlrun ci demo:

```yaml
kind: job
metadata:
  name: remote-git-test
  project: default
  tag: latest
spec:
  command: 'myfunc.py'
  args: []
  image_pull_policy: Always
  build:
    commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/ci-demo.git
``` 

## Working with Archive

If you work with local files and want mlrun to imporporate all of those into your function container
you can use the archive option, it will tar your working directory and upload it into an archive path
the remote builder will untar all the files into the container working directory

example, create a `function.yaml` file in the working directory with the following text:

```yaml
kind: job
metadata:
  name: remote-demo4
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image_pull_policy: Always
  build:
    commands: ['pip install mlrun pandas']
    base_image: python:3.6-jessie
```

the function spec tells the system to use a python base image and add couple of packages
, the `examples/training.py` will be the file we execute on `run` commands.

next we build that function (`.` is a shortcut for `function.yaml`).
we use the `-a` flag to specify we want to tar and upload our local dir to the remote archive 
(`,/` is the default source, another source can be specified using `-s` options).
The builder will inflate the tar into the container working dir.

    mlrun build . -a v3io:///users/admin/tars
    
once the function is built we can run it with some parameters

    mlrun run -f . -w -p p1=3