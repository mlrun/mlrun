# Installation and setup guide <!-- omit in toc -->

This guide lists the prerpequisites, and outlines the steps for installing and running MLRun. 

## Overview

- Install the MLRun service [locally using Docker](https://docs.mlrun.org/en/latest/install/local-docker.html) or [over Kubernetes Cluster](https://docs.mlrun.org/en/latest/install/kubernetes.html). Alternatively, you can use [Iguazio's managed MLRun service](https://www.iguazio.com/docs/latest-release/).
- [Set up your client environment](https://docs.mlrun.org/en/latest/install/remote.html) to work with the local or remote MLRun service.

Once you have installed and configured MLRun, follow the [Quick Start tutorial](https://docs.mlrun.org/en/latest/tutorial/01-mlrun-basics.html) and additional [Tutorials and Examples](https://docs.mlrun.org/en/latest/tutorial/index.html) to learn how to use MLRun to develop and deploy machine 
learning applications to production.

For interactive installation and usage tutorials, try the [MLRun Katakoda Scenarios](https://www.katacoda.com/mlrun).

**Installation options:**
```{toctree}
:maxdepth: 1

install/remote
install/local-docker
install/kubernetes
```

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