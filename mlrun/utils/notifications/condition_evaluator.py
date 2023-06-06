import multiprocessing

import mlrun.model


def evaluate_notification_condition_in_separate_process(
    run: mlrun.model.RunObject, notification: mlrun.model.Notification, timeout: int = 5
):
    receiver, sender = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=_evaluate_notification_condition_wrapper,
        args=(sender, run, notification),
    )
    p.start()
    if receiver.poll(timeout):
        result = receiver.recv()
        p.join()
        return result
    else:
        p.kill()
        raise TimeoutError(f"Condition evaluation timed out after {timeout} seconds")


def _evaluate_notification_condition_wrapper(
    connection, run: mlrun.model.RunObject, notification: mlrun.model.Notification
):
    connection.send(_evaluate_notification_condition(run, notification))
    return connection.close()


def _evaluate_notification_condition(
    run: mlrun.model.RunObject, notification: mlrun.model.Notification
):
    if not notification.condition:
        return True

    import jinja2.sandbox

    jinja_env = jinja2.sandbox.SandboxedEnvironment()
    template = jinja_env.from_string(notification.condition)
    result = template.render(run=run.to_dict(), notification=notification.to_dict())
    if result.lower() in ["0", "no", "n", "f", "false"]:
        return False

    # if the condition is not a boolean, we assume we need to send the notification anyway
    return True
