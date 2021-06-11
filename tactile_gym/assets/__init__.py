import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_assets_path():
    return _ROOT


def add_assets_path(path):
    return os.path.join(_ROOT, path)
