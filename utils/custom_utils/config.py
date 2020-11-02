import json
from dotmap import DotMap
import os
import time
import argparse
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    # config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "logs/")
    # config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "checkpoints/")
    return config


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--config",
        dest="config",
        metavar="C",
        default="None",
        help="The Configuration file",
    )
    args = argparser.parse_args()
    return args
