from os import environ
from threading import Lock

import yaml

import viaenv

env_prefix = 'MLRUN'
env_file_key = f'{env_prefix}_CONIFG_FILE'
_load_lock = Lock()
_loaded = False


# Default values & types
class Config:
    namespace: str = 'default-tenant'


# Global configuration
config = Config()


def populate():
    """Populate configuration from config file (if exists in environment) and
    from environment variables.

    populate will run only once, after first call it does nothing.
    """
    global _loaded

    with _load_lock:
        if _loaded:
            return
        _populate(config)
        _loaded = True


def _populate(config):
    config_path = environ.get(env_file_key)
    if config_path:
        with open(config_path) as fp:
            config_data = yaml.safe_load(fp)

        if not isinstance(config_data, dict):
            raise TypeError(f'configuration in {config_path} not a dict')

        for name, cls in config.__annotations__.items():
            if name in config_data:
                value = config_data[name]
                if not isinstance(value, cls):
                    typ = type(value).__name__
                    raise TypeError(
                        f'{config_path}:{name!r} - bad type, '
                        f' wanted {cls.__name__}, got {typ}'
                    )
                setattr(config, name, value)

    viaenv.populate_from_env(config, prefix=env_prefix)
