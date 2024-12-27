# -*- coding: utf-8 -*-

import os
import random
import shutil
from time import time
from datetime import datetime
import configparser
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.cuda.amp import autocast as autocast

from lib.utils_model_pre1 import *
from lib.mydataset import MyDataset
from model.model_pre1 import Net_block as model
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
FLAGS = parser.parse_args()
decay = FLAGS.decay
num_nodes = 50
epochs = 200
batch_size= FLAGS.batch_size
points_per_5min = 1
num_for_predict = 1
num_of_days = 1
num_of_hours = 3
num_of_5min = 12

merge = False
model_name = 'BILSTM12'
params_dir = 'experiment'
prediction_path = 'BILSTM_prediction'
wdecay = 0.000

device = torch.device(FLAGS.device)
print('read matrix')
# read matrix
adj_mx_list=[]
adj1 = './data/array_50_100.pkl'
adj_mx1 = load_graph_data_hz(adj1)
print(adj_mx1)
for i in range(len(adj_mx1)):
    adj_mx1[i, i] = 0
adj_mx_list.append(adj_mx1)

adj_mx = np.stack(adj_mx_list, axis=-1)
# print(adj_mx.shape)
adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)
src, dst = adj_mx.sum(axis=-1).nonzero()

edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                         dtype=torch.float,
                         device=device)
print(edge_index)
print(edge_attr.shape)
print(edge_index.shape)

# Metro_edge_matrix = np.load('./data/all_5min.npy')[:,:,:]# 8760，69，69
# print(Metro_edge_matrix.shape)
# Metro_week_matrix = np.load('./data/ext2018_week_Matrix.npy')  # [(45,56),[]]
# Metro_hour_matrix = np.load('./data/ext2018_hour_Matrix.npy')  #
Metro_week_matrix = None
Metro_hour_matrix = None




print('Model is %s' % (model_name))

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    seed_torch(12)
    print('12')
    # read all data from graph signal matrix file
    print("Reading data...")
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    f=open('data/scaler.txt', 'r')
    s=eval(f.readline())
    f.close()
    scaler = StandardScaler(mean=np.array(s['mean']), std=np.array(s['std']))
    print('scaler,mean: %.6f,  std: %.6f' % (scaler.mean, scaler.std))

    # test set ground truth
    true_value = np.load('data/train_target.npy')
    true_val_value = np.load('data/val_target.npy')

    train_dataset = MyDataset('data/train_day.npy', 'data/train_hour.npy','data/train_recent.npy','data/train_target.npy')
    val_dataset = MyDataset('data/val_day.npy', 'data/val_hour.npy','data/val_recent.npy','data/val_target.npy')
    test_dataset = MyDataset('data/test_day.npy', 'data/test_hour.npy','data/test_recent.npy','data/test_target.npy')
    # training set data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # validation set data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # testing set data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # loss function MSE
    loss_function = nn.MSELoss()

    # get model's structure
    net = model(device,edge_index,edge_attr)

    net.to(device)  # to cuda
    scaler = scaler
    scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)

    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)


    his_mae = []
    train_time = []
    SUM = len(train_loader)/batch_size
    count = 0
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_d, train_h, train_r, train_t in train_loader:
            train_d = train_d.to(device)
            train_h = train_h.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0
            with autocast():
                output = net([train_d, train_h, train_r],
                            [])
                # output = scaler_torch.inverse_transform(output)
                # train_t = scaler_torch.inverse_transform(train_t)


                loss = loss_function(output, train_t)

            loss.backward()


            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)

            # print(count)
            # count+=FLAGS.batch_size
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.5f, time: %.5fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)


        valid_loss, val_mae, val_rmse = compute_val_loss(net, val_loader,true_val_value, loss_function,device, epoch,scaler)

        his_mae.append(val_mae)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                                  epoch, str(round(val_mae, 4))))
        os.makedirs(params_path, exist_ok=True)
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))

    bestid = np.argmin(his_mae)

    print("The valid loss on best model is epoch%s_%s" % (str(bestid + 1), str(round(his_mae[bestid], 4))))
    best_params_filename = os.path.join(params_path,
                                        '%s_epoch_%s_%s.params' % (model_name,
                                                                   str(bestid + 1), str(round(his_mae[bestid], 4))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction= predict(net, test_loader,device)

    end_time_test = time()
    evaluate(net, test_loader, true_value, device, epoch,scaler)
    test_time = np.mean(end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)
