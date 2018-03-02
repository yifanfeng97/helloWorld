import sys
import os
import os.path as osp
import ConfigParser

class config():
    def __init__(self, cfg_file='config/config.cfg'):

        cfg = ConfigParser.SafeConfigParser()
        cfg.read(cfg_file)
        #default
        self.base_dir = cfg.get('DEFAULT', 'base_dir')
        self.result_dir = cfg.get('DEFAULT', 'result_dir')
        self.resume_training = cfg.getboolean('DEFAULT', 'resume_training')
        # cls
        self.modelnet_cls_dir = cfg.get('CLS', 'modelnet_cls_dir')

        self.check_dirs()

    def check_dirs(self):
        self.check_dir(self.result_dir)

    def check_dir(self, dir):
        if not osp.exists(dir):
            os.mkdir(dir)