from base64 import b64decode
from http import HTTPStatus

from flask import jsonify, request

from ..config import config
from .app import app

basic_prefix = 'Basic '
bearer_prefix = 'Bearer '


class AuthError(Exception):
    pass


def basic_auth_required(cfg):
    return cfg.user or cfg.password


def bearer_auth_required(cfg):
    return cfg.token


def parse_basic_auth(header):
    """
    >>> parse_basic_auth('Basic YnVnczpidW5ueQ==')
    ['bugs', 'bunny']
    """
    b64value = header[len(basic_prefix):]
    value = b64decode(b64value).decode()
    return value.split(':', 1)


@app.before_request
def check_auth():
    if request.path == '/api/healthz':
        return

    cfg = config.httpdb

    header = request.headers.get('Authorization', '')
    try:
        if basic_auth_required(cfg):
            if not header.startswith(basic_prefix):
                raise AuthError('missing basic auth')
            user, passwd = parse_basic_auth(header)
            if user != cfg.user or passwd != cfg.password:
                raise AuthError('bad basic auth')
        elif bearer_auth_required(cfg):
            if not header.startswith(bearer_prefix):
                raise AuthError('missing bearer auth')
            token = header[len(bearer_prefix):]
            if token != cfg.token:
                raise AuthError('bad bearer auth')
    except AuthError as err:
        resp = jsonify(ok=False, error=str(err))
        resp.status_code = HTTPStatus.UNAUTHORIZED
        return resp
