import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


def data_path(*parts):
    return os.path.join(ROOT_DIR, "data", *parts)


def temp_path(*parts):
    return os.path.join(ROOT_DIR, "temp", *parts)


def mkdir_recursive(dir):
    parts = []

    while not os.path.exists(dir):
        dir, part = os.path.split(dir)
        parts.append(part)

    for part in parts[::-1]:
        dir = os.path.join(dir, part)
        os.mkdir(dir)
