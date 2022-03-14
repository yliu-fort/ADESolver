import json
from pathlib import Path


def clear_output_dir(dir):
    [f.unlink() for f in Path(dir).glob("*") if f.is_file()]
    return True


def load_input(input_meta):
    with open(input_meta, ) as handle:
        return json.load(handle)