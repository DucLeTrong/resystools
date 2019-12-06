import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from resystools.implicite import model as mod
from resystools.implicite import config
from resystools.implicite import evaluate
from resystools.implicite import data_utils

def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer 

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_test(model_name='NeuMF-end',lr=0.001, dropout=0, batch_size=256, epochs=50, top_k=10,
            factor_num=32, num_layers=3, num_ng=4, test_num_ng=99, out=True, gpu="0"):
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    torch.backends.cudnn.deterministic=True
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cudnn.benchmark = True
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, num_ng, True)
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
            batch_size=test_num_ng+1, shuffle=False, num_workers=0)

    print("Data is loaded!")
########################### CREATE MODEL #################################
    if model_name == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = mod.NCF(user_num, item_num, factor_num, num_layers, 
                            dropout, model_name, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()

    if model_name == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    # path_log = "log_training_" + config.model+ str(time.time())

    count, best_hr = 0, 0
    for epoch in range(epochs):
        # if epoch < 10:
        #     optimizer = scheduler(optimizer,0.001)
        # elif epoch < 13:
        #     optimizer = scheduler(optimizer,0.0001)
        # elif epoch < 15:
        #     optimizer = scheduler(optimizer,0.00002)
        # else:
        #     optimizer = scheduler(optimizer,0.000005)
        
        # print("epoch: ", epoch," ------ learning rate", get_lr(optimizer))
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, top_k)
        elapsed_time = time.time() - start_time

        # with open(path_log, "a") as fi:
        #     fi.write(str(epoch) +" "+ str(time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))+" "+str(np.mean(HR)) +" "+ str(np.mean(NDCG)))
        # print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
        #         time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        # print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(model, 
                    '{}{}.pth'.format(config.model_path, config.model))

        print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg))
