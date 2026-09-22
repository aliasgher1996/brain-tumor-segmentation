import argparse
import copy
import os

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, 'configs', 'default.yaml')


class Config(dict):
    """dict with attribute access (cfg.train.epochs)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


def load_config(path=DEFAULT_CONFIG, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError('unknown config key: {}'.format(key))
        node[leaf] = yaml.safe_load(value)
    return Config(copy.deepcopy(cfg))


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    parser.add_argument('--opts', nargs='*', default=[],
                        help='overrides such as train.epochs=50 data.root=/path/to/kaggle_3m')
    return parser
