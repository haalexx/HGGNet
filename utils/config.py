import yaml
from easydict import EasyDict


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(args):
    data_config = cfg_from_yaml_file(args.data)
    model_config = cfg_from_yaml_file(args.cfg)
    return data_config, model_config


