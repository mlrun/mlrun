(notifications)=

# Notifications

MLRun supports configuring notifications on jobs and scheduled jobs. This section describes the SDK for notifications.

- [The notification object](#the-notification-object)
- [Local vs. remote](#local-vs-remote)
- [Notification parameters and secrets](#notification-params-and-secrets)
- [Notification kinds](#notification-kinds)
- [Configuring notifications for runs](#configuring-notifications-for-runs)
- [Configuring notifications for pipelines](#configuring-notifications-for-pipelines)
- [Setting notifications on live runs](#setting-notifications-on-live-runs)
- [Setting notifications on scheduled runs](#setting-notifications-on-scheduled-runs)
- [Notification conditions](#notification-conditions)


## The notification object
The notification object's schema is:
- `kind`: str - notification kind (slack, git, etc...)
- `when`: list[str] - run states on which to send the notification (completed, error, running)
- `name`: str - notification name
- `message`: str - notification message
- `severity`: str - notification severity (info, warning, error, debug)
- `params`: dict - notification parameters (See definitions in [Notification Kinds](#notification-params-and-secrets))
- `secret_params`: dict - secret data notification parameters (See definitions in [Notification Params and Secrets](#notification-kinds))
- `condition`: str - jinja template for a condition that determines whether the notification is sent or not (See [Notification Conditions](#notification-conditions))


## Local vs. remote
Notifications can be sent either locally from the SDK, or remotely from the MLRun API. 
Usually, a local run sends locally, and a remote run sends remotely.
However, there are several special cases where the notification is sent locally either way.
These cases are:
- Local or KFP Engine Pipelines: To conserve backwards compatibility, the SDK sends the notifications as it did before adding the run
  notifications mechanism. This means you need to watch the pipeline in order for its notifications to be sent. (Remote pipelines act differently. See [Configuring Notifications For Pipelines](#configuring-notifications-for-pipelines For Pipelines for more details.)
- Dask: Dask runs are always local (against a remote Dask cluster), so the notifications are sent locally as well.

> **Disclaimer:** Notifications of local runs aren't persisted.

## Notification parameters and secrets
The notification parameters often contain sensitive information, such as Slack webhooks Git tokens, etc.
To ensure the safety of this sensitive data, the parameters are split into 2 objects - `params` and `secret_params`.
Either can be used to store any notification parameter. However the `secret_params` will be protected by project secrets.
When a notification is created, its `secret_params` are automatically masked and stored in a mlrun project secret.
The name of the secret is built from the hash of the params themselves (So if multiple notifications use the same secret, it won't waste space in the project secret).
Inside the notification's `secret_params`, you'll find a reference to the secret under the `secret` key once it's been masked.
For non-sensitive notification parameters, you can simply use the `params` parameter, which doesn't go through this masking process.
It's essential to utilize `secret_params` exclusively for handling sensitive information, ensuring secure data management.


## Notification kinds

Currently, the supported notification kinds and their params are as follows:

- `slack`:
  - `webhook`: The slack webhook to which to send the notification.
- `git`:
  - `token`: The git token to use for the git notification.
  - `repo`: The git repo to which to send the notification.
  - `issue`: The git issue to which to send the notification.
  - `merge_request`: In GitLab (as opposed to GitHub), merge requests and issues are separate entities. 
                     If using merge request, the issue will be ignored, and vice versa.
  - `server`: The git server to which to send the notification.
  - `gitlab`: (bool) Whether the git server is GitLab or not.
- `webhook`:
  - `url`: The webhook url to which to send the notification.
  - `method`: The http method to use when sending the notification (GET, POST, PUT, etc...).
  - `headers`: (dict) The http headers to send with the notification.
  - `override_body`: (dict) The body to send with the notification. If not specified, the body will be a dict with the 
                     `name`, `message`, `severity`, and the `runs` list of the completed runs. You can also add the run's details using: `"override_body": {"message":"Run Completed {{ runs }}"`.
					 Results would look like 
					 ```
					 {
                       "message": "Run Completed [{'project': 'my-project', 'name': 'my-function', 'host': <run-host>, 'status': {'state': 'completed', 'results': <run-results>}}]"
                     }
					 ```
  - `verify_ssl`: (bool) Whether SSL certificates are validated during HTTP requests or not,
                  The default is set to `True`.
- `console` (no params, local only)
- `ipython` (no params, local only)

## Configuring notifications for runs

In any `run` method you can configure the notifications via their model. For example:

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

MLRun can also send a `pipeline started` notification. To do that, configure a notification that includes
`when=running`. The `pipeline started` notification uses its own parameters, for
example the webhook, credentials, etc., for the notification message.

If the webhook for the running notification is stored in the secret_params, you should first set the project secret
and then use this project secret in the notification. For example:
```python
import mlrun

project = mlrun.get_or_create_project("ycvqowgpie")
project.set_secrets({"SLACK_SECRET1": '{"webhook":"<WEBHOOK_URL>"}'})
slack_notification = mlrun.model.Notification(
    kind="slack",
    when=["running"],
    name="name",
    message="message",
    condition="",
    severity="verbose",
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
