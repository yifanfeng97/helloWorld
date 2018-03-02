import torch
import models.ClsNet
import os
import sys

def get_model(cfg, dataset='modelnet40', resume=False, mode='cls'):
    if dataset=='modelnet40':
        class_num = 40
    else:
        class_num = 55
    if mode=='cls':
        model = models.ClsNet.ClsNet(1, class_num)

    model.cuda()
    return model

def save_checkpoint(model, output_path):
    ## if not os.path.exists(output_dir):
    ##    os.makedirs("model/")
    torch.save(model, output_path)

    print("Checkpoint saved to {}".format(output_path))

def save_model_and_optim(model, optimizer, epoch,
                         best_prec1, best_confusion_mat,
                         model_checkpoint, optim_checkpoint):
    checkpoint = {}
    if isinstance(model, torch.nn.DataParallel):
        model_save = model.module
    elif isinstance(model, torch.nn.Module):
        model_save = model
    else:
        print('model save type error')
        sys.exit()
    checkpoint['model_param'] = model_save.cpu().state_dict()
    model_save.cuda()

    save_checkpoint(checkpoint, model_checkpoint)

    optim_state = {}
    optim_state['epoch'] = epoch  # because epoch starts from 0
    optim_state['best_prec1'] = best_prec1
    optim_state['optim_state_best'] = optimizer.state_dict()
    optim_state['best_confusion_matrix'] = best_confusion_mat
    save_checkpoint(optim_state, optim_checkpoint)
    # problem, should we store latest optim state or model, currently, we donot