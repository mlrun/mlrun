# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def version():
    with open('mlrun/__init__.py') as fp:
        for line in fp:
            if '__version__' in line:
                _, version = line.split('=')
                return version.replace("'", '').strip()


def load_deps(section):
    """Load dependencies from Pipfile, we can't assume toml is installed"""
    # [packages]
    header = '[{}]'.format(section)
    with open('Pipfile') as fp:
        in_section = False
        for line in fp:
            line = line.strip()
            if not line or line[0] == '#':
                continue

            if line == header:
                in_section = True
                continue

            if line.startswith('['):
                in_section = False
                continue

            if in_section:
                # ipython = ">=6.5"
                i = line.find('=')
                assert i != -1, 'bad dependency - {}'.format(line)
                pkg = line[:i].strip()
                version = line[i+1:].strip().replace('"', '')
                if version == '*':
                    yield pkg
                else:
                    yield '{}{}'.format(pkg, version.replace('"', ''))


with open('README.md') as fp:
    long_desc = fp.read()

install_requires = list(load_deps('packages'))
tests_require = list(load_deps('dev-packages'))


setup(
    name='mlrun',
    version=version(),
    description='Tracking and config of machine learning runs',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    author='Yaron Haviv',
    author_email='yaronh@iguazio.com',
    license='MIT',
    url='https://github.com/mlrun/mlrun',
    packages=['mlrun', 'mlrun.runtimes', 'mlrun.db', 'mlrun.platforms'],
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    tests_require=tests_require,
    zip_safe=False,
    include_package_data=True,
    entry_points={'console_scripts': [
        'mlrun=mlrun.__main__:main']},
)
