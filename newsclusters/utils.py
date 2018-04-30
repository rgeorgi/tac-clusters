"""
A handful of convenience functions.
"""

import yaml

def text_or_none(item):
    if item is None:
        return None
    else:
        return item.text.strip()


def load_config(path: str) -> yaml.YAMLObject:
    """
    Load the config file.
    """
    with open(path, 'r') as y:
        return yaml.load(y)