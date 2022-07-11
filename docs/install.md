# Installation and setup guide <!-- omit in toc -->

This guide outlines the steps for installing and running MLRun. 

MLRun comprises of two parts: MLRun Server and MLRUN client.

## Deployment options


There are several deployment options:
- [Local deployment](https://docs.mlrun.org/en/latest/install/local-docker.html): Deploy a Docker on your laptop or on a single server.
   This option is good for testing the waters or when working in a small scale environment. It's limited in terms of computing resources and scale, but
   simpler for deployment.

- [Kubernetes cluster](https://docs.mlrun.org/en/latest/install/kubernetes.html): Deploy an MLRun server on Kubernetes.
   This option deploys MLRun on a Kubernetes cluster, which supports elastic scaling. Yet, it is more complex to install as it requires you to install Kubernetes on your own.
  
- [Iguazio's Managed  Service](https://www.iguazio.com): A commerical offering by Iguazio. This is the fastest way to explore the full set of MLRun functionalities.<br>
  Note that Iguazio provides a 14 day free trial.


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
