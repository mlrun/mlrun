import mlrun.projects
from mlrun.__main__ import add_notification_to_project, line2keylist
from mlrun.utils import list2dict


def test_add_notification_to_cli_from_file():
    notification = ("file=notification",)
    notifications = line2keylist(notification, keyname="type", valname="params")
    print(notifications)
    project = mlrun.projects.MlrunProject(name="test")
    for notification in notifications:
        if notification["type"] == "file":
            with open(notification["params"]) as fp:
                lines = fp.read().splitlines()
                notification = list2dict(lines)
                add_notification_to_project(notification, project)

        else:
            add_notification_to_project(
                {notification["type"]: notification["params"]}, project
            )

    assert project._notifiers._notifications["slack"].params.get('webhook') == "123234"
    assert project._notifiers._notifications["ipython"].params.get('webhook') == "1232"


def test_add_notification_to_cli_from_dict():
    notification = ('slack={"webhook":"123234"}', 'ipython={"webhook":"1232"}')
    notifications = line2keylist(notification, keyname="type", valname="params")
    print(notifications)
    project = mlrun.projects.MlrunProject(name="test")
    for notification in notifications:
        if notification["type"] == "file":
            with open(notification["params"]) as fp:
                lines = fp.read().splitlines()
                notification = list2dict(lines)
                add_notification_to_project(notification, project)

        else:
            add_notification_to_project(
                {notification["type"]: notification["params"]}, project
            )

    assert project._notifiers._notifications["slack"].params.get('webhook') == "123234"
    assert project._notifiers._notifications["ipython"].params.get('webhook') == "1232"
