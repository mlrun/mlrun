(notifications)=

# Notifications
Notifications are used to inform you of system events on jobs, both scheduled and manually triggered.
For regular jobs (manual and scheduled), you can receive notifications when the job finishes (`completed`, `error`, or `aborted`).
For workflows (e.g., MLRun pipelines), you can also receive notifications when the  job starts (`running`).

This section describes the notifications SDK and its usage.

**In this section**

- [SDK](#sdk)
- [Local vs. remote](#local-vs-remote)
- [Notification parameters and secrets](#notification-parameters-and-secrets)
- [MLRun on Iguazio](#mlrun-on-iguazio)
- [MLRun CE](#mlrun-ce)
- [Mail notifications](#mail-notifications)
- [Configuring notifications for runs](#configuring-notifications-for-runs)
- [Configuring notifications for pipelines](#configuring-notifications-for-pipelines)
- [Setting notifications on live runs](#setting-notifications-on-live-runs)
- [Setting notifications on scheduled runs](#setting-notifications-on-scheduled-runs)
- [Notification conditions](#notification-conditions)


## SDK
- {py:class}`~mlrun.common.schemas.notification.Notification`: The notification object
- {py:class}`~mlrun.common.schemas.notification.NotificationKind`: The notification kinds
- {py:class}`~mlrun.db.httpdb.HTTPRunDB.refresh_smtp_configuration`: Gets or refreshes the SMTP configuration from the Iguazio platform and sets it
as the default SMTP configuration (creates an `mlrun-smtp-config` with the SMTP configuration). For privileged user: `IT Admin`.


## Local vs. remote
Notifications can be sent either locally from the SDK, or remotely from the MLRun API. 
Usually, a local run sends locally, and a remote run sends remotely.
However, there are several special cases where the notification is sent locally either way.
These cases are:
- Local: To conserve backwards compatibility, the SDK sends the notifications as it did before adding the run
  notifications mechanism. This means you need to watch the pipeline in order for its notifications to be sent. (Remote pipelines act differently. See [Configuring Notifications For Pipelines](#configuring-notifications-for-pipelines) for more details.
- Dask: Dask runs are always local (against a remote Dask cluster), so the notifications are sent locally as well.

> **Disclaimer:** Notifications of local runs aren't persisted.

## Notification parameters and secrets
The notification parameters often contain sensitive information, such as Slack webhooks, Git tokens, etc.
To ensure the safety of this sensitive data, the parameters are split into two objects - `params` and `secret_params`.
Either can be used to store any notification parameter. However the `secret_params` is protected by project secrets.
When a notification is created, its `secret_params` are automatically masked and stored in an mlrun project secret.
The name of the secret is built from the hash of the parameters themselves (so if multiple notifications use the same secret, it doesn't waste space in the project secret).
Inside the notification's `secret_params`, there's a reference to the secret under the `secret` key after it's masked.
For non-sensitive notification parameters, you can simply use the `params` parameter, which doesn't go through this masking process.
It's essential to utilize `secret_params` exclusively for handling sensitive information, ensuring secure data management.


## Mail notifications
### MLRun on Iguazio
When MLRun is deployed on the Iguazio platform, your IT Admin can [configure the SMTP server in the Iguazio platform](https://www.iguazio.com/docs/latest-release/cluster-mgmt/deployment/post-deployment-howtos/smtp/). To use this as your default configuration, run the following (with privileged user - `IT Admin`):
```python
import mlrun

mlrun.get_run_db().refresh_smtp_configuration()
```
The `refresh_smtp_configuration` method gets the SMTP configuration from the Iguazio platform and sets it
as the default, cluster-wide, SMTP configuration (creates an `mlrun-smtp-config` secret with the SMTP configuration).
If you edit the configuration on the Iguazio platform, run the `refresh_smtp_configuration` method again.

Three parameters cannot be configured on the Iguazio platform. To set their defaults for the cluster, run these commands with the relevant values:
```
kubectl -n default-tenant patch secret mlrun-smtp-config -p='{"stringData":{"use_tls":"false"}}'
kubectl -n default-tenant patch secret mlrun-smtp-config -p='{"stringData":{"start_tls":"true"}}'
kubectl -n default-tenant patch secret mlrun-smtp-config -p='{"stringData":{"validate_certs":"false"}}'
```
These parameters are maintained when modifying parameters on the Iguazio platform.

### MLRun CE
In the community edition, you can use your own SMTP server.
To configure it, manually create the `mlrun-smtp-config` Kubernetes secret with the default
parameters for the SMTP server (`server_host`, `server_port`, `username`, `password`, `start_tls`, etc.).
After creating or editing the secret, refresh the MLRun SMTP configuration by running the `refresh_smtp_configuration` method.

### Create a mail notification object

The following snippet shows the format of a notification object.
You can inherit the default `mlrun-smtp-config`, or choose to overwrite parameter/s.
Any `params` not defined in this format will be enriched with the values in the `mlrun-smtp-config` secret.
The only mandatory field in `params` is `email_addresses`.

```python
mail_notification = mlrun.model.Notification(
    kind="mail",
    when=["completed", "error", "running"],
    name="mail-notification",
    message="",
    condition="",
    severity="verbose",
    params={
        "email_addresses": ["user.name@domain.com"],
    },
)
```
MLRun uses the [aiosmtplib](https://aiosmtplib.readthedocs.io/en/stable/) library for sending mail notifications.
The `params` argument is a dictionary that supports the following fields:
 - `server_host` (string): The SMTP server host
 - `server_port` (int): The SMTP server port
 - `sender_address` (string): The sender email address
 - `username` (string): The username for the SMTP server
 - `password` (string): The password for the SMTP server
 - `email_addresses` (list of strings): The list of email addresses to send the mail to
 - `start_tls` (boolean): Whether to start the TLS connection
 - `use_tls` (boolean): Whether to use TLS
 - `validate_certs` (boolean): Whether to validate the certificates

You can read more about `start_tls` and `use_tls` on the  [aiosmtplib docs](https://aiosmtplib.readthedocs.io/en/stable/encryption.html).

Email notifications on local runs must explicitly include all SMTP settings.

## Configuring notifications for runs

In any `run` method you can configure the notifications via their model. For example:

```python
notification = mlrun.model.Notification(
    kind="webhook",
    when=["completed", "error"],
    name="notification-1",
    message="completed",
    severity="info",
    secret_params={"url": "<webhook url>"},
    params={"method": "GET", "verify_ssl": True},
)
function.run(handler=handler, notifications=[notification])
```
To add run details to the notification:
```python
notifications_func = [
    mlrun.model.Notification.from_dict(
        {
            "kind": "webhook",
            "name": "Test",
            "severity": "info",
            "when": ["error", "completed"],
            "condition": "",
            "params": {
                "url": webhook_test,
                "method": "POST",
                "override_body": {"message": "Run Completed {{ runs }}"},
            },
        }
    ),
]
```

The results look like:
```
{
  "message": "Run Completed [{'project': 'test-remote-workflow', 'name': 'func-func', 'host': 'func-func-pkt97', 'status': {'state': 'completed', 'results': {'return': 1}}}]"
}
```


## Configuring notifications for pipelines
To set notifications on pipelines, supply the notifications in the run method of either the project or the pipeline.
For example:
```python
notification = mlrun.model.Notification(
    kind="webhook",
    when=["completed", "error", "running"],
    name="notification-1",
    message="completed",
    severity="info",
    secret_params={"url": "<webhook url>"},
    params={"method": "GET", "verify_ssl": True},
)
project.run(..., notifications=[notification])
```

### Running notifications
MLRun can also send a `pipeline started` notification for KFP pipelines (and not for job runs). To do that, configure a notification that includes
`when=running`. The `pipeline started` notification uses its own parameters, for
example the webhook, credentials, etc., for the notification message.
You can set only the webhook; the message is the default message.

If the webhook is stored in the secret_params, you should first set the project secret and then use this project secret
in the notification. For example:
```python
import mlrun

project = mlrun.get_or_create_project("ycvqowgpie")
project.set_secrets({"SLACK_SECRET1": '{"webhook":"<WEBHOOK_URL>"}'})
slack_notification = mlrun.model.Notification(
    kind="slack",
    when=["running"],
    name="name",
    condition="",
    secret_params={"secret": "SLACK_SECRET1"},
)
```

### Remote pipeline notifications
In remote pipelines, the pipeline end notifications are sent from the MLRun API. This means you don't need to watch the pipeline in order for its notifications to be sent.
The pipeline start notification is still sent from the SDK when triggering the pipeline.

### Local and KFP engine pipeline notifications
In these engines, the notifications are sent locally from the SDK. This means you need to watch the pipeline in order for its notifications to be sent.
This is a fallback to the old notification behavior, therefore not all of the new notification features are supported. Only the notification kind and params are taken into account.
In these engines the old way of setting project notifiers is still supported:

```python
project.notifiers.add_notification(
    notification_type="slack", params={"webhook": "<slack webhook url>"}
)
project.notifiers.add_notification(
    notification_type="git",
    params={"repo": "<repo>", "issue": "<issue>", "token": "<token>"},
)
```
Instead of passing the webhook in the notification `params`, it is also possible in a Jupyter notebook to use the ` %env` 
magic command:
```
%env SLACK_WEBHOOK=<slack webhook url>
```

Editing and removing notifications is done similarly with the following methods:
```python
project.notifiers.edit_notification(
    notification_type="slack", params={"webhook": "<new slack webhook url>"}
)
project.notifiers.remove_notification(notification_type="slack")
```

## Setting notifications on live runs
You can set notifications on live runs via the `set_run_notifications` method. For example:

```python
import mlrun

mlrun.get_run_db().set_run_notifications(
    "<project-name>", "<run-uid>", [notification1, notification2]
)
```

Using the `set_run_notifications` method overrides any existing notifications on the run. To delete all notifications, pass an empty list.

## Setting notifications on scheduled runs
You can set notifications on scheduled runs via the `set_schedule_notifications` method. For example:

```python
import mlrun

mlrun.get_run_db().set_schedule_notifications(
    "<project-name>", "<schedule-name>", [notification1, notification2]
)
```

Using the `set_schedule_notifications` method overrides any existing notifications on the schedule. To delete all notifications, pass an empty list.

## Notification conditions
You can configure the notification to be sent only if the run meets certain conditions. This is done using the `condition`
parameter in the notification object. The condition is a string that is evaluated using a jinja templator with the run 
object in its context. The jinja template should return a boolean value that determines whether the notification is sent or not. 
If any other value is returned or if the template is malformed, the condition is ignored and the notification is sent 
as normal.

Take the case of a run that calculates and outputs model drift. This example code sets a notification to fire only
if the drift is above a certain threshold:

```python
notification = mlrun.model.Notification(
    kind="slack",
    when=["completed", "error"],
    name="notification-1",
    message="completed",
    severity="info",
    secret_params={"webhook": "<slack webhook url>"},
    condition='{{ run["status"]["results"]["drift"] > 0.1 }}',
)
```
