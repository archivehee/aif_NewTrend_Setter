import os
import pickle
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_ndcg20 = -1.0
    best_epoch = -1
    no_improve_cnt = 0
    patience = 10  # number of validations without improvement
    # index of k=20 in world.topks (fallback: last index)
    try:
        ndcg20_idx = list(world.topks).index(20)
    except ValueError:
        ndcg20_idx = len(world.topks) - 1

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
        # validation every 5 epochs based on NDCG@20
        if (epoch + 1) % 5 == 0:
            cprint("[VALIDATION]")
            val_dict = getattr(dataset, "validDict", None)
            val_results = Procedure.Test(dataset,
                                         Recmodel,
                                         epoch,
                                         w,
                                         world.config['multicore'],
                                         eval_dict=val_dict,
                                         split_name="valid")
            current_ndcg20 = float(val_results['ndcg'][ndcg20_idx])
            if current_ndcg20 > best_ndcg20:
                best_ndcg20 = current_ndcg20
                best_epoch = epoch
                no_improve_cnt = 0
                best_state_dict = Recmodel.state_dict()
                cprint(f"[NEW BEST] epoch {epoch+1}, NDCG@20={best_ndcg20:.6f}")
            else:
                no_improve_cnt += 1
                cprint(f"[NO IMPROVE] {no_improve_cnt}/{patience} validations")
                if no_improve_cnt >= patience:
                    cprint(f"[EARLY STOP] no NDCG@20 improvement for {patience} validations")
                    break
finally:
    if world.tensorboard:
        w.close()
    # load best model (if found) before final evaluation
    try:
        if 'best_state_dict' in locals():
            Recmodel.load_state_dict(best_state_dict)
            cprint(f"[LOAD BEST] epoch {best_epoch+1} with NDCG@20={best_ndcg20:.6f}")
    except Exception as e:
        print(f"failed to load best state dict: {e}")
    # final evaluation on test split at best epoch
    cprint("[FINAL TEST]")
    Procedure.Test(dataset,
                   Recmodel,
                   world.TRAIN_epochs,
                   None,
                   world.config['multicore'],
                   eval_dict=getattr(dataset, "testDict", None),
                   split_name="test")
    if world.SAVE_PRETRAIN:
        save_dir = world.PRETRAIN_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"user_{world.dataset}.pkl")
        user_emb = Recmodel.embedding_user.weight.detach().cpu().numpy()
        with open(save_path, "wb") as fout:
            pickle.dump(user_emb, fout)
        cprint(f"[SAVE USER EMB] {save_path}")
