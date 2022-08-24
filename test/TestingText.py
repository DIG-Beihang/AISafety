#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('../')
sys.path.append('./')
import argparse
from pathlib import Path

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__" and __package__ is None:
    level = 1
    # https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
    import importlib

    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass
    __package__ = ".".join(parent.parts[len(top.parts) :])
    importlib.import_module(__package__)  # won't be needed after that

from cli_new import main as cli_main


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description="AI-Testing Text Module",
    )
    parser.add_argument(
        "config_file_path",
        nargs=1,
        type=str,
        help="Config file (.yml or .yaml file) path",
    )

    args = vars(parser.parse_args())

    config_file_path = Path(args["config_file_path"][0])

    return config_file_path


def main(config_file_path):
    """ """
    config_file_path = Path(config_file_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file {config_file_path} not found")
    if config_file_path.suffix not in [".yml", ".yaml"]:
        raise ValueError(f"Config file {config_file_path} must be a .yml or .yaml file")
    config = yaml.safe_load(config_file_path.read_text())

    # print(config)

    cli_main(config)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
