import yaml


def load_config():
    config_file = open('./configs/configs.yaml', mode='r')
    return yaml.load(config_file, Loader=yaml.FullLoader)
