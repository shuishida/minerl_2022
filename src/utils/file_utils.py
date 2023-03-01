import os
import json

import yaml


class JSONIO(dict):
    def __init__(self, path=None):
        self.filepath = path
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                super(JSONIO, self).__init__(json.load(f))
        else:
            super(JSONIO, self).__init__()

    def __setitem__(self, key, value):
        super(JSONIO, self).__setitem__(str(key), value)
        if self.filepath:
            with open(self.filepath, "w") as f:
                json.dump(self, f)

    def __getitem__(self, index):
        return super(JSONIO, self).__getitem__(str(index))


def load_yaml(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config
