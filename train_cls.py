from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel  # for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import train_helper
import os
from utils import meter
from utils import config
from data_helper import point_datasets
import numpy as np


# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

def train(train_loader, model, criterion, optimizer, epoch, cfg):
    """
    train for one epoch on the training set
    """
    batch_time = meter.timemeter.TimeMeter()
    data_time = meter.timemeter.TimeMeter()
    losses = meter.averagevaluemeter.AverageValueMeter()
    prec = meter.classerrormeter.ClassErrorMeter(topk=[1], accuracy=True)

    # training mode
    model.train()

    for i, (inputs_img, gt_labels) in enumerate(train_loader):
        batch_time.reset()
        if isinstance(inputs_img, torch.ByteTensor):
            inputs_img = inputs_img.float()
        gt_labels = gt_labels.long().view(-1)
        inputs_img = Variable(inputs_img)
        gt_labels = Variable(gt_labels)

        # shift data to GPU
        inputs_img = inputs_img.cuda()
        gt_labels = gt_labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        preds = model(inputs_img)  # bz x C x H x W
        preds = preds.squeeze()

        softmax_loss = criterion(preds, gt_labels)
        loss = softmax_loss

        prec.add(preds.data, gt_labels.data)
        losses.add(loss.data[0], preds.size(0))  # batchsize

        optimizer.zero_grad()
        loss.backward()
        # train_helper.clip_gradient(optimizer, cfg.gradient_clip)
        optimizer.step()

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Loss {loss:.4f} \t'
                  'Prec@1 {top1:.3f}\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), loss=losses.value()[0], top1=prec.value(1)))

            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(train_loader) + i
            # loss_name= None
            # if cfg.have_aux:
            #     loss_name = 'mixed_loss'
            # else:
            #     loss_name = 'loss'
            # info = { loss_name : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'train/' + tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     if not hasattr(value.grad, 'data'): continue
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'train/' + tag, to_np(value), step)
            #     logger.histo_summary(log_pre_name+'train/' + tag + '/grad', to_np(value.grad), step)
            # # images
    print('mean class accuracy at epoch {0}: \t'
          'top1:{1}\t'.format(epoch, prec.value(1)))


def validate(val_loader, model, criterion, epoch, confusion_mat, cfg):
    """
    validation for one epoch on the validation set
    """
    batch_time = meter.timemeter.TimeMeter()
    data_time = meter.timemeter.TimeMeter()
    losses = meter.averagevaluemeter.AverageValueMeter()
    prec = meter.classerrormeter.ClassErrorMeter(topk=[1], accuracy=True)
    if confusion_mat is None:
        confusion_mat = meter.confusionmeter.ConfusionMeter(cfg.num_classes, normalized=True)

    # evaluate mode
    model.eval()

    for i, (input_img, gt_labels) in enumerate(val_loader):
        batch_time.reset()

        if isinstance(input_img, torch.ByteTensor):
            input_img = input_img.float()
        gt_labels = gt_labels.long().view(-1)
        input_img = Variable(input_img, volatile=True)
        gt_labels = Variable(gt_labels, volatile=True)

        # shift data to GPU
        input_img = input_img.cuda()
        gt_labels = gt_labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        preds = model(input_img)  # bz x C x H x W
        preds = preds.squeeze()

        loss = criterion(preds, gt_labels)

        prec.add(preds.data, gt_labels.data)
        losses.add(loss.data[0], preds.size(0))  # batchsize
        confusion_mat.add(preds.data, gt_labels.data)

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Loss {loss:.4f} \t'
                  'Prec@1 {top1:.3f}\t'.format(
                epoch, i, len(val_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), loss=losses.value()[0], top1=prec.value(1)))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(test_loader) + i
            # info = { 'loss' : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'test/'+tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'test/'+tag, to_np(value), step)
            # # images

    print('mean class accuracy at epoch {0}: \t'
          'top1:{1}\t'.format(epoch, prec.value(1)))
    print('confusion matrix:')
    print(confusion_mat.value())

    return prec.value(1), confusion_mat.value()


def main():
    cfg = config.config()

    best_prec1 = 0
    # only used when we resume training from some checkpoint model
    resume_epoch = 0

    if cfg.resume_training and os.path.exists(cfg.modelnet_init_cls_model_file):
        model = train_helper.get_model(cfg, dataset='modelnet40', resume=True)
    else:
        model = train_helper.get_model(cfg, dataset='modelnet40')

    print('model: ')
    print(model)

    # multiple gpu
    # model.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), cfg.lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)

    # if we load model from pretrained, we need the optim state here
    if cfg.resume_training and os.path.exists(cfg.modelnet_init_cls_optim_file):
        print('loading optim epoch prec from {0}'.format(cfg.modelnet_init_cls_optim_file))
        optim_state = torch.load(cfg.modelnet_init_cls_optim_file)

        resume_epoch = optim_state['epoch'] + 1
        best_prec1 = optim_state['best_prec1']
        best_confusion_mat = optim_state['best_confusion_matrix']
        optimizer.load_state_dict(optim_state['optim_state_best'])
        del optim_state

    criterion = nn.CrossEntropyLoss()

    print('shift model and criterion to GPU .. ')
    # model = model.cuda()
    # define loss function (criterion) and pptimizer
    criterion = criterion.cuda()

    train_loader = point_datasets.point_modelnet40_Dataset_cls(mode='train')
    val_loader = point_datasets.point_modelnet40_Dataset_cls(mode='test')

    for epoch in range(resume_epoch, cfg.max_epoch):

        train(train_loader, model, criterion, optimizer, epoch, cfg)
        prec1, confusion_mat = validate(val_loader, model, criterion, epoch, cfg)

        if best_prec1 < prec1:
            # save checkpoints
            best_prec1 = prec1
            best_confusion_mat = confusion_mat
            train_helper.save_model_and_optim(cfg, model, optimizer, epoch, best_prec1, best_confusion_mat)

        print('best accuracy: ', best_prec1)
        print('best confusion matrix:')
        print(best_confusion_mat)


if __name__ == '__main__':
    main()