# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .imptriplet_loss import ImpTripletLoss
from .retriplet_loss import ReTripletLoss
from .center_loss import CenterLoss 
from .crossentropylabelsmooth import CrossEntropyLabelSmooth

def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    imptriplet = ImpTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    retriplet = ReTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=cfg.DATASETS.NUM_CLASSES) #label smooth
 
    if 'center' in sampler:
        center_criterion = CenterLoss(num_classes=cfg.DATASETS.NUM_CLASSES, feat_dim=cfg.MODEL.FEAT_DIM, use_gpu=True)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    elif sampler == 'softmax_imptriplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + imptriplet(feat, target)[0]
    elif sampler == 'softmax_retriplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + retriplet(feat, target)[0]
    elif sampler == 'softmax_triplet_center':
        #Warm up learning rate
        #Random erasing augmentation
        #Label smoothing
        #Last stride 1
        #BNNeck
        #Center loss
        def loss_func(score, feat, target):
            return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_imptriplet, '
              'but got {}'.format(sampler))
    if 'center' in sampler:
        return loss_func, center_criterion
    else:
        return loss_func
