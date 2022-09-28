(install-setup-guide)=
# Installation and setup guide <!-- omit in toc -->

This guide outlines the steps for installing and running MLRun. 

MLRun comprises two parts: MLRun Server and MLRUN client.

**In this section**
- [Deployment options](#deployment-options)
- [Non-root user support](#non-root-user-support)
- [Security context](#security-context)
- [Set up your client](#Set-up-your-client)
- [MLRun client backward compatibility](#MLRun-client-backward-compatibility)

## Deployment options

There are several deployment options:
- [Local deployment](https://docs.mlrun.org/en/latest/install/local-docker.html): Deploy a Docker on your laptop or on a single server.
   This option is good for testing the waters or when working in a small scale environment. It's limited in terms of computing resources 
   and scale, but simpler for deployment.
- [Kubernetes cluster](https://docs.mlrun.org/en/latest/install/kubernetes.html): Deploy an MLRun server on Kubernetes.
   This option deploys MLRun on a Kubernetes cluster, which supports elastic scaling. Yet, it is more complex to install as it requires 
   you to install Kubernetes on your own.
- [Iguazio's Managed  Service](https://www.iguazio.com): A commercial offering by Iguazio. This is the fastest way to explore 
  the full set of MLRun functionalities.<br>
  Note that Iguazio provides a 14 day free trial.
  
## Non-root user support

By default, MLRun assigns the root user to MLRun runtimes and pods. You can improve the security context by changing the security mode, 
which is implemented by Igauzio during installation, and applied system-wide:
- Override: Use the user id of the user that triggered the current run or use the nogroupid for group id. Requires Iguazio v3.5.1.
- Disabled: Security context is not auto applied (the system aplies the root user). (default)

## Security context

If your system is configured in disabled mode, you can apply the security context to individual runtimes/pods by using `function.with_security_context`, and the job is assigned to the user or to the user's group that ran the job.<br>
(You cannot override the user of individual jobs if the system is configured in override mode.) The options are:

```
from kubernetes import client as k8s_client

security_context = k8s_client.V1SecurityContext(
            run_as_user=1000,
            run_as_group=3000,
        )
function.with_security_context(security_context)
```
See the [full definition of the V1SecurityContext object](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1SecurityContext.md).

Some services do not support security context yet:
- Infrastructure services
   - Kubeflow pipelines core services
- Services created by MLRun
   - Kaniko, used for building images. (To avoid using Kaniko, use prebuilt images that contain all the requirements.) 
   - Spark services

## Set up your client

- You can work with your favorite IDE (e.g. Pycharm, VScode, Jupyter , Colab etc..). Read how to configure your client against the deployed
MLRun server in [How to configure your client](https://docs.mlrun.org/en/latest/install/remote.html).

Once you have installed and configured MLRun, follow the [Quick Start tutorial](https://docs.mlrun.org/en/latest/tutorial/01-mlrun-basics.html) and additional [Tutorials and Examples](https://docs.mlrun.org/en/latest/tutorial/index.html) to learn how to use MLRun to develop and deploy machine 
learning applications to production.

For interactive installation and usage tutorials, try the [MLRun Katakoda Scenarios](https://www.katacoda.com/mlrun).


<a id="MLRun-client-backward-compatibility"></a>
## MLRun client backward compatibility  

Starting from MLRun 0.10.0, the MLRun client and images are compatible with minor MLRun releases that are released during the following 6 months. When you upgrade to 0.11.0, for example, you can continue to use your 0.10-based images. 

```{admonition} Important
- Images from 0.9.0 are not compatible with 0.10.0. Backward compatibility starts from 0.10.0. 
- When you upgrade the MLRun major version, for example 0.10.x to 1.0.x, there is no backward compatibility. 
- The feature store is not backward compatible. 
- When you upgrade the platform, for example from 3.2 to 3.3, the clients should be upgraded. There is no guaranteed compatibility with an older MLRun client after a platform upgrade. 
```

See also [Images and their usage in MLRun](https://docs.mlrun.org/en/latest/runtimes/images.html#mlrun-images-and-how-to-build-them).

```{toctree}
:hidden:
:maxdepth: 1

install/local-docker
install/kubernetes
install/remote
```
