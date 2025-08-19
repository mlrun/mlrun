(install-setup-guide)=
# Installation and setup guide <!-- omit in toc -->

This guide outlines the steps for installing the MLRun service and client and for running MLRun. 

MLRun has two main components, the service and the client (SDK and UI):

- The MLRun service can orchestrate and integrate with other open source frameworks, as shown in the following diagram. 
- The MLRun client SDK is installed in your development environment and interacts with the service using REST API calls. 

This release of MLRun supports only Python 3.9 for both the server and the client. 

<p align="center"><img src="_static/images/mlrun-cluster.png" alt="mlrun-flow" width="700"/></p><br>

**In this section**
- [Deployment options](#deployment-options)
- [Set up your client](#set-up-your-client)
- [Security](#security)

## Deployment options

The deployment options are:
- {ref}`Kubernetes<install-on-kubernetes>`: Deploys the MLRun CE server over Kubernetes. 
- {ref}`AWS cluster<aws-install>`: Deploys the MLRun CE server on an AWS cluster.
- [Iguazio's Managed  Service](https://www.iguazio.com): A commercial offering by Iguazio. This is the fastest way to explore the full set of MLRun functionalities.<br>
  Note that Iguazio provides a 14 day free trial.

You can also deploy the MLRun Service using local Docker for demo and test purposes.

## Set up your client

You can work with your favorite IDE (e.g. PyCharm, VSCode, Jupyter, Colab, etc.). Read how to configure your client against the deployed
MLRun server in {ref}`install-remote`.

Once you have installed and configured MLRun, follow the [Quick Start tutorial](./tutorials/01-mlrun-basics.ipynb) and additional {ref}`Tutorials and Examples<tutorial>` to learn how to use MLRun to develop and deploy machine learning applications to production.


<a id="MLRun-client-backward-compatibility"></a>
### MLRun client backward compatibility  

Starting from MLRun v1.3.0, the MLRun server is compatible with the client and images of the previous two minor MLRun releases. When you upgrade to v1.3.0, for example, you can continue to use your v1.1- and v1.2-based images, but v1.0-based images are not compatible.

After you update the MLRun package client version by running `pip install mlrun==<"new-client-version">`, you must update the images to use the same client version you installed.
For example, when running this command `pip install mlrun==1.8.0` you must update your images to use MLRun v1.8.0 by adding `mlrun==<"new-client-version">` as a function requirement. See {py:meth}`~mlrun.runtimes.BaseRuntime.with_requirements`.

```{admonition} Important
- Images from 0.9.0 are not compatible with 0.10.0. Backward compatibility starts from 0.10.0. 
- When you upgrade the MLRun major version, for example 0.10.x to 1.0.x, there is no backward compatibility. 
- The feature store is not backward compatible. 
- When you upgrade the platform, for example from 3.2 to 3.3, the clients should be upgraded. There is no guaranteed compatibility with an older MLRun client after a platform upgrade. 
```

See also {ref}`images-usage`.


## Security

### Non-root user support

By default, MLRun assigns the root user to MLRun runtimes and pods. You can improve the security context by changing the security mode, 
which is implemented by Iguazio during installation, and applied system-wide:
- Override: Use the user id of the user that triggered the current run or use the `nogroupid` for group id. Requires Iguazio v3.5.1.
- Disabled: Security context is not auto applied (the system applies the root user). (default)


<br>
<img class="dark-light" src="_static/images/maintenance_logo.svg" alt="Maintenance logo" width="250"/>

```{toctree}
:hidden:
:maxdepth: 1

install/kubernetes
install/aws-install
install/remote
```
