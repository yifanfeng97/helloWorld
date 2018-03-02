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
        self.modelnet_cls_result_dir = cfg.get('CLS', 'modelnet_cls_result_dir')
        self.modelnet_init_cls_dir = cfg.get('CLS', 'modelnet_init_cls_dir')
        self.modelnet_init_cls_model_file = cfg.get('CLS', 'modelnet_init_cls_model_file')
        self.modelnet_init_cls_optim_file = cfg.get('CLS', 'modelnet_init_cls_optim_file')
        # train
        self.lr = cfg.getfloat('TRAIN', 'lr')
        self.momentum = cfg.getfloat('TRAIN', 'momentum')
        self.weight_decay = cfg.getfloat('TRAIN', 'weight_decay')
        self.max_epoch = cfg.getfloat('TRAIN', 'max_epoch')

        self.check_dirs()

    def check_dirs(self):
        self.check_dir(self.result_dir)
        self.check_dir(self.modelnet_cls_result_dir)
        self.check_dir(self.modelnet_init_cls_dir)

    def check_dir(self, dir):
        if not osp.exists(dir):
            os.mkdir(dir)