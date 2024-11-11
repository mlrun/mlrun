import pytest

import mlrun.utils.notifications.notification
from mlrun.utils import logger

DEFAULT_PARAMS = {
    "server_host": "smtp.gmail.com",
    "server_port": 587,
    "sender_address": "sender@example.com",
    "username": "user",
    "password": "pass",
    "email_addresses": "a@example.com",
    "use_tls": True,
}


class TestMailNotification:
    @pytest.mark.parametrize(
        "params, should_raise",
        [
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": "a@example.com",
                },
                False,
            ),
            (
                {
                    "server_host": "smtp.gmail.com",
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": ["a@example.com", "b@example.com"],
                },
                False,
            ),
            (
                {
                    "server_port": 587,
                    "sender_address": "sender@example.com",
                    "username": "user",
                    "password": "pass",
                    "email_addresses": "a@example.com",
                },
                True,
            ),
        ],
    )
    def test_validate_mail_params(self, params, should_raise):
        try:
            mlrun.utils.notifications.notification.MailNotification.validate_params(
                params
            )
        except ValueError:
            assert should_raise
        else:
            assert not should_raise

    @pytest.mark.parametrize(
        ["name", "params", "expected_params"],
        [
            (
                "missing_all_params",
                {},
                DEFAULT_PARAMS,
            ),
            (
                "overriding_some_params",
                {
                    "server_host": "another@smtp.com",
                    "server_port": 589,
                },
                {
                    "server_host": "another@smtp.com",
                    "server_port": 589,
                },
            ),
            (
                "email_addresses_as_list",
                {
                    "email_addresses": ["a@b.com", "b@b.com", "c@c.com"],
                },
                {"email_addresses": "a@b.com,b@b.com,c@c.com"},
            ),
        ],
    )
    def test_enrich_default_params(self, name, params, expected_params):
        logger.debug(f"Testing {name}")
        enriched_params = mlrun.utils.notifications.notification.MailNotification.enrich_default_params(
            params, DEFAULT_PARAMS
        )
        default_params_copy = DEFAULT_PARAMS.copy()
        default_params_copy.update(expected_params)
        assert enriched_params == default_params_copy
