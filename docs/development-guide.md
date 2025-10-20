(Development Guide)=
# Development Guide <!-- omit in toc -->

Essential information for MLRun developers and operators
Learn about MLRun deployment options and version compatibility requirements.

**In this section**
- [Deployment options](#deployment-options)
- [MLRun client backward compatibility](#MLRun-client-backward-compatibility)

## Deployment options

The deployment options are:
- {ref}`Kubernetes<install-on-kubernetes>`: Deploys the MLRun CE server over Kubernetes. 
- {ref}`AWS cluster<aws-install>`: Deploys the MLRun CE server on an AWS cluster.
- [Iguazio's Managed  Service](https://www.iguazio.com): A commercial offering by Iguazio. This is the fastest way to explore the full set of MLRun functionalities.<br>
  Note that Iguazio provides a 14 day free trial.

(MLRun-client-backward-compatibility)=
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


<br>
<img class="dark-light" src="_static/images/maintenance_logo.svg" alt="Maintenance logo" width="250"/>

```{toctree}
:hidden:
:maxdepth: 1


/development-guide/remote.md
```
