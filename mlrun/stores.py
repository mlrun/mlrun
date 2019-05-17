# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import zipfile
from base64 import b64encode
import yaml
import requests
from os import path, remove, environ
import shlex
from argparse import ArgumentParser
import boto3
from urllib.parse import urlparse, ParseResult
from shutil import copyfile


def build_zip(zip_path, files=[]):
    z = zipfile.ZipFile(zip_path, "w")
    for f in files:
        if not path.isfile(f):
            raise Exception('file name {} not found'.format(f))
        z.write(f)
    z.close()


def unzip(zip_path, files=[]):
    files_data = {}
    with zipfile.ZipFile(zip_path) as myzip:
        for f in files:
            with io.TextIOWrapper(myzip.open(f)) as zipped:
                files_data[f] = zipped.read()
    return files_data


def upload_file(file_path, url, del_file=False):
    url2repo(url).upload(file_path)
    if del_file:
        remove(file_path)


def put_data(url, data):
    url2repo(url).put(data)


def url2repo(url='', secrets={}):
    if '://' not in url:
        return FileRepo(url)
    p = urlparse(url)
    scheme = p.scheme.lower()
    if scheme == 's3':
        return S3Repo(p, secrets)
    elif scheme == 'git':
        return GitRepo(p, secrets)
    elif scheme == 'http' or scheme == 'https':
        return HttpRepo(p, secrets)
    elif scheme == 'v3io' or scheme == 'v3ios':
        return V3ioRepo(p, secrets)
    else:
        raise ValueError('unsupported repo scheme ({})'.format(scheme))


class ExternalRepo:
    def __init__(self, urlobj: ParseResult):
        self.urlobj = urlobj
        self.kind = ''

    def get(self):
        pass

    def put(self, data):
        pass

    def download(self, target_path):
        pass

    def upload(self, src_path):
        pass


class FileRepo(ExternalRepo):
    def __init__(self, path=''):
        self.path = path
        self.kind = 'file'

    def get(self):
        with open(self.path, 'r') as fp:
            return fp.read()

    def put(self, data):
        with open(self.path, 'w') as fp:
            fp.write(data)
            fp.close()

    def download(self, target_path):
        copyfile(self.path, target_path)

    def upload(self, src_path):
        copyfile(src_path, self.path)


class S3Repo(ExternalRepo):
    def __init__(self, urlobj: ParseResult, secrets={}):
        self.kind = 's3'
        self.bucket = urlobj.hostname
        self.key = urlobj.path[1:]
        region = None

        access_key = urlobj.username or secrets.get('AWS_ACCESS_KEY_ID')
        secret_key = urlobj.password or secrets.get('AWS_SECRET_ACCESS_KEY')

        if access_key or secret_key:
            self.s3 = boto3.resource('s3', region_name=region,
                                     aws_access_key_id=access_key,
                                     aws_secret_access_key=secret_key)
        else:
            # from env variables
            self.s3 = boto3.resource('s3', region_name=region)

    def upload(self, src_path):
        self.s3.Object(self.bucket, self.key).put(Body=open(src_path, 'rb'))

    def get(self):
        obj = self.s3.Object(self.bucket, self.key)
        return obj.get()['Body'].read()

    def put(self, data):
        self.s3.Object(self.bucket, self.key).put(Body=data)


def basic_auth_header(user, password):
    username = user.encode('latin1')
    password = password.encode('latin1')
    base = b64encode(b':'.join((username, password))).strip()
    authstr = 'Basic ' + base.decode('ascii')
    return {'Authorization': authstr}


def http_get(url, headers=None, auth=None):
    try:
        resp = requests.get(url, headers=headers, auth=auth)
    except OSError:
        raise OSError('error: cannot connect to {}'.format(url))

    if not resp.ok:
        raise OSError('failed to read file in {}'.format(url))
    return resp.text


def http_put(url, data, headers=None, auth=None):
    try:
        resp = requests.put(url, data=data, headers=headers, auth=auth)
    except OSError:
        raise OSError('error: cannot connect to {}'.format(url))
    if not resp.ok:
        raise OSError(
            'failed to upload to {} {}'.format(url, resp.status_code))


def http_upload(url, file_path, headers=None, auth=None):
    with open(file_path, 'rb') as data:
        http_put(url, data, headers, auth)


class HttpRepo(ExternalRepo):
    def __init__(self, urlobj: ParseResult, secrets={}):
        self.kind = 'http'
        host = urlobj.hostname
        if urlobj.port:
            host += ':{}'.format(urlobj.port)
        self.url = '{}://{}{}'.format(urlobj.scheme, host, urlobj.path)
        if urlobj.username or urlobj.password:
            self.auth = (urlobj.username, urlobj.password)
        else:
            self.auth = None

    def upload(self, src_path):
        raise ValueError('unimplemented')

    def put(self, data):
        raise ValueError('unimplemented')

    def get(self):
        return http_get(self.url, None, self.auth)


class V3ioRepo(ExternalRepo):
    def __init__(self, urlobj: ParseResult, secrets={}):
        self.kind = 'v3io'
        host = urlobj.hostname or environ.get('V3IO_API')
        if urlobj.port:
            host += ':{}'.format(urlobj.port)
        self.url = 'http://{}{}'.format(host, urlobj.path)

        token = environ.get('V3IO_ACCESS_KEY')
        username = urlobj.username or secrets.get('V3IO_USERNAME') or environ.get('V3IO_USERNAME')
        password = urlobj.password or secrets.get('V3IO_PASSWORD') or environ.get('V3IO_PASSWORD')

        self.headers = None
        self.auth = None
        if (not urlobj.username and urlobj.password) or token:
            token = urlobj.password or token
            self.headers = {'X-v3io-session-key': token}
        elif username and password:
            self.headers = basic_auth_header(username, password)

    def upload(self, src_path):
        http_upload(self.url, src_path, self.headers, None)

    def get(self):
        return http_get(self.url, self.headers, None)

    def put(self, data):
        http_put(self.url, data, self.headers, None)


class GitRepo(ExternalRepo):
    def __init__(self, urlobj: ParseResult, secrets={}):
        self.kind = 'git'
        host = urlobj.hostname or 'github.com'
        if urlobj.port:
            host += ':{}'.format(urlobj.port)
        self.path = 'https://{}{}'.format(host, urlobj.path)

        self.headers = {'Authorization': ''}
        token = urlobj.username or environ.get('GIT_ACCESS_TOKEN')
        if token:
            self.headers = {'Authorization': 'token '.format(token)}

        # format: git://[token@]github.com/org/repo#master[:<workdir>]
        self.branch = 'master'
        self.workdir = None
        if urlobj.fragment:
            parts = urlobj.fragment.split(':')
            if parts[0]:
                self.branch = parts[0]
            if len(parts) > 1:
                self.workdir = parts[1]

    def upload(self, src_path):
        raise ValueError('unimplemented, use git push instead')

    def get(self):
        raise ValueError('unimplemented, use git pull instead')

    def put(self, data):
        raise ValueError('unimplemented, use git push instead')
