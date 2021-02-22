from tests.api.runtime_handlers.test_kubejob import TestKubejobRuntimeHandler


class TestRemoteSparkjobRuntimeHandler(TestKubejobRuntimeHandler):
    """
    Remote Spark runtime behaving pretty much like the kubejob runtime just with few modifications (several automations
    we want to do for the user) so we're simply running the same tests as the ones of the job runtime
    """

    def _get_class_name(self):
        return "remote-spark"
