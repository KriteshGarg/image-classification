import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Config(object):

    def __init__(self, config_path='./config/config.yml'):
        logger.info("loading config...")
        with open(config_path, "r") as config_file:
            self.config_yml = yaml.safe_load(config_file)

        # add attributes to Config object as per config.yml file
        tags = dict()
        for key, value in self.config_yml.items():
            setattr(self, key, value)
            tags[key] = value
            logger.info(f"Config - {key} : {value}")
