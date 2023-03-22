(Function_storage_auto_mount)=
# Attach storage to functions

In the vast majority of cases, an MLRun function requires access to storage. This storage
might be used to provide inputs to the function including data-sets to process or data-streams that contain input events.
Typically, storage is used to store function outputs and result artifacts. For example, trained models or processed
data-sets.

Since MLRun functions can be distributed and executed in Kubernetes pods, the storage used would typically be shared, 
and execution pods would need some added configuration options applied to them so that the function code is able to 
access the designated storage. These configurations might be k8s volume mounts, specific environment variables that 
contain configuration and credentials, and other configuration of security settings. These storage 
configurations are not applicable to functions running locally in the development environment, since they are executed 
in the local context.

The common types of shared storage are:

1. `v3io` storage through API &mdash; When running as part of the Iguazio system, MLRun has access to the system's v3io
storage through paths such as `v3io:///projects/my_projects/file.csv`. To enable this type of access, several
environment variables need to be configured in the pod that provide the v3io API URL and access keys.
2. `v3io` storage through FUSE mount &mdash; Some tools cannot utilize the `v3io` API to access it and need basic filesystem
semantics. For that purpose, `v3io` provides a FUSE (Filesystem in user-space) driver that can be used to mount `v3io` 
containers as specific paths in the pod itself. For example `/User`. To enable this, several specific volume mount 
configurations need to be applied to the pod spec.
3. NFS storage access &mdash; When MLRun is deployed as open-source, independent of Iguazio, the deployment automatically adds
a pod running NFS storage. To access this NFS storage through pods, a kubernetes `pvc` mount is needed.
4. Others &mdash; As use-cases evolve, other cases of storage access may be needed. This will require various configurations 
to be applied to function execution pods.

MLRun attempts to offload this storage configuration task from the user by automatically applying the most common 
storage configuration to functions. As a result, most cases do not require any additional storage configurations 
before executing a function as a Kubernetes pod. The configurations applied by MLRun are:

* In an Iguazio system, apply configurations for `v3io` access through the API.
* In an open-source deployment where NFS is configured, apply configurations for `pvc` access to NFS storage.

This MLRun logic is referred to as **auto-mount**.

**In this section**
- [Disabling auto-mount](#disabling-auto-mount)
- [Modifying the auto-mount default configuration](#Modifying-the-auto-mount-default-configuration)

## Disabling auto-mount
In cases where the default storage configuration does not fit the function needs, MLRun allows for function spec 
modifiers to be manually applied to functions. These modifiers can add various configurations to the function spec, 
adding environment variables, mounts and additional configurations. MLRun also provides a set of common modifiers 
that can be used to apply storage configurations.
These modifiers can be applied by using the `.apply()` method on the function and adding the modifier to apply. 
You can see some examples of this later in this page.

When a different storage configuration is manually applied to a function, MLRun's auto-mount logic is disabled. This 
prevents conflicts between configurations. The auto-mount logic can also be disabled by setting
`func.spec.disable_auto_mount = True` 
on any MLRun function. 

## Modifying the auto-mount default configuration
The default auto-mount behavior applied by MLRun is controlled by setting MLRun configuration parameters. 
For example, the logic can be set to automatically mount the `v3io` FUSE driver on all functions, or perform `pvc` 
mount for NFS storage on all functions.
The following code demonstrates how to apply the `v3io` FUSE driver by default:

    # Change MLRun auto-mount configuration
    import mlrun.mlconf

    mlrun.mlconf.storage.auto_mount_type = "v3io_fuse"

Each of the auto-mount supported methods applies a specific modifier function. The supported methods are:
* `v3io_credentials` &mdash; apply `v3io` credentials needed for `v3io` API usage. Applies the 
{py:meth}`~mlrun.platforms.v3io_cred` modifier.
* `v3io_fuse` &mdash; create Fuse driver mount. Applies the {py:meth}`~mlrun.platforms.mount_v3io` modifier.
* `pvc` &mdash; create a `pvc` mount. Applies the {py:meth}`~mlrun.platforms.mount_pvc` modifier.
* `auto` &mdash; the default auto-mount logic as described above (either `v3io_credentials` or `pvc`).
* `none` &mdash; perform no auto-mount (same as using `disable_auto_mount = True`).

The modifier functions executed by auto-mount can be further configured by specifying their parameters. These can be 
provided in the `storage.auto_mount_params` configuration parameters. Parameters can be passed as a string made of 
`key=value` pairs separated by commas. For example, the following code runs a `pvc` mount with specific parameters:

    mlrun.mlconf.storage.auto_mount_type = "pvc"
    pvc_params = {
        "pvc_name": "my_pvc_mount",
        "volume_name": "pvc_volume",
        "volume_mount_path": "/mnt/storage/nfs",
    }
    mlrun.mlconf.storage.auto_mount_params = ",".join(
        [f"{key}={value}" for key, value in pvc_params.items()]
    )

Alternatively, the parameters can be provided as a base64-encoded JSON object, which can be useful when passing complex
parameters or strings that contain special characters:

    pvc_params_str = base64.b64encode(json.dumps(pvc_params).encode())
    mlrun.mlconf.storage.auto_mount_params = pvc_params_str

