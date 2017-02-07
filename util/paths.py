import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


def data_path(*parts):
    return os.path.join(ROOT_DIR, "data", *parts)


def temp_path(*parts):
    return os.path.join(ROOT_DIR, "temp", *parts)