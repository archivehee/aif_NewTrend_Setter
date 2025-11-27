'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    hr, ndcg = [], []
    for k in world.topks:
        hr.append(utils.HR_at_K(groundTrue, r, k))
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        'hr': np.array(hr),
        'ndcg': np.array(ndcg)
    }
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0, eval_dict=None, split_name="test"):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = eval_dict if eval_dict is not None else dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    results = {
        'hr': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks))
    }
    num_users_eval = 0
    num_neg = 99
    with torch.no_grad():
        users = list(testDict.keys())
        if len(users) == 0:
            print(f"[{split_name.upper()}] no users to evaluate, skip.")
            return results
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating_full = Recmodel.getUsersRating(batch_users_gpu)
            rating_full = rating_full.cpu().numpy()  # [batch_size, m_items]

            for idx, u in enumerate(batch_users):
                # ground truth item (leave-one-out, so use the first)
                gt_items = testDict.get(u, [])
                if len(gt_items) == 0:
                    continue
                gt = gt_items[0]

                # positive items from training for this user
                pos_items = set(dataset.getUserPosItems([u])[0])
                pos_items.add(gt)

                # sample 99 negatives that are not positive
                negs = []
                while len(negs) < num_neg:
                    neg = np.random.randint(0, dataset.m_items)
                    if neg not in pos_items:
                        negs.append(neg)
                        pos_items.add(neg)  # avoid duplicates

                candidates = [gt] + negs
                scores = rating_full[idx, candidates]
                order = np.argsort(-scores)
                ranked = np.array(candidates)[order]

                num_users_eval += 1
                for ki, K in enumerate(world.topks):
                    topk_items = ranked[:K]
                    if gt in topk_items:
                        results['hr'][ki] += 1.0
                        pos = np.where(topk_items == gt)[0][0]
                        results['ndcg'][ki] += 1.0 / np.log2(pos + 2)
                    # else: add zero

        if num_users_eval > 0:
            results['hr'] /= float(num_users_eval)
            results['ndcg'] /= float(num_users_eval)

        if world.tensorboard and w is not None:
            w.add_scalars(f'Test/HR@{world.topks}',
                          {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        print(f"[{split_name.upper()}] {results}")
        return results
