import os


ROOT_DIR = os.path.dirname(__file__)

def add_rootpath(path):
    return os.path.join(ROOT_DIR, path)