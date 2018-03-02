import sys
import os
import ConfigParser

class config():
    def __init__(self, cfg_file='config/config.cfg'):

        cfg = ConfigParser.SafeConfigParser()
        cfg.read(cfg_file)