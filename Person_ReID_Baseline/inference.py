import fire
import os
import time
import torch
import numpy as np
from torch.autograd import Variable
import models
from config import cfg
from data_loader import data_loader
from logger import make_logger
from evaluation import evaluation
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run
import matplotlib.pyplot as plt

if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

    
def test(config_file, **kwargs):
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()
    
    re_ranking=cfg.RE_RANKING
    
    PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    if not re_ranking:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,'result')
        logger.info("Test Results:")
    else:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,'result_re-ranking')
        logger.info("Re-Ranking Test Results:") 
    
    device = torch.device(cfg.DEVICE)
    
    _, val_loader, num_query, num_classes = data_loader(cfg,cfg.DATASETS.NAMES)
    
    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    model.load(cfg.OUTPUT_DIR,cfg.TEST.LOAD_EPOCH)
    if device:
        model.to(device) 
    model = model.eval()

    all_feats = []
    all_pids = []
    all_camids = []
    all_imgs = []
    
    for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
        with torch.no_grad():
            images, pids, camids = data
            all_imgs.extend(images.numpy())
            if device:
                model.to(device) 
                images = images.to(device)
            
            feats = model(images)

        all_feats.append(feats)
        all_pids.extend(np.asarray(pids))
        all_camids.extend(np.asarray(camids))

    all_feats = torch.cat(all_feats, dim=0)
    # query
    qf = all_feats[:num_query]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    q_imgs = all_imgs[:num_query]
    # gallery
    gf = all_feats[num_query:]
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])
    g_imgs = all_imgs[num_query:]

    if not re_ranking::
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
    else:
        print('Calculating Distance')
        q_g_dist = np.dot(qf.data.cpu(), np.transpose(gf.data.cpu()))
        q_q_dist = np.dot(qf.data.cpu(), np.transpose(qf.data.cpu()))
        g_g_dist = np.dot(gf.data.cpu(), np.transpose(gf.data.cpu()))
        print('Re-ranking:')
        distmat= re_ranking(q_g_dist, q_q_dist, g_g_dist)

    indices = np.argsort(distmat, axis=1)

    mean=cfg.INPUT.PIXEL_MEAN
    std=cfg.INPUT.PIXEL_STD
    top_k = 7
    for i in range(num_query):
        # get query pid and camid
        q_pid = q_pids[i]
        q_camid = q_camids[i]

        # remove gallery samples that have the same pid and camid with query
        order = indices[i]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # binary vector, positions with value 1 are correct matches
        true_index = indices[i][keep]

        plt.title("top5 query",fontsize=15)
        plt.subplot(181)
        img = np.clip(q_imgs[i].transpose(1,2,0)*std+mean,0.0,1.0)
        plt.imshow(img)
        for j in range(top_k):
            plt.subplot(182+j)
            img = np.clip(g_imgs[true_index[j]].transpose(1,2,0)*std+mean,0.0,1.0)
            plt.imshow(img)
        plt.savefig("./show/{}.jpg".format(i))
            
    logger.info('Testing complete')
    
if __name__=='__main__':
    fire.Fire(test)
