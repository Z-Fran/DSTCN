import torch
import numpy as np
from lib.utils_model_pre1 import *
from lib.mydataset import MyDataset
from model.model_pre1 import Net_block
from torch.utils.data import DataLoader

device = 'cuda:0'
batch_size = 1

true_val_value = np.load('data/val_target.npy')
print(true_val_value)

print('read matrix')
# read matrix
adj_mx_list=[]
adj1 = './data/all_adj.pkl'
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

# get model's structure
net = Net_block(device,edge_index,edge_attr)
net.load_state_dict(torch.load('experiment/BILSTM12/BILSTM12_epoch_166_0.011.params'))
net.to(device)  # to cuda
net.eval()


train_dataset = MyDataset('data/train_day.npy', 'data/train_hour.npy','data/train_recent.npy','data/train_target.npy')
val_dataset = MyDataset('data/val_day.npy', 'data/val_hour.npy','data/val_recent.npy','data/val_target.npy')
test_dataset = MyDataset('data/test_day.npy', 'data/test_hour.npy','data/test_recent.npy','data/test_target.npy')
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
f=open('data/scaler.txt', 'r')
s=eval(f.readline())
f.close()
scaler = StandardScaler(mean=np.array(s['mean']), std=np.array(s['std']))
scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)

with torch.no_grad():
    prediction = []
    for index, (test_w, test_d, test_r, test_t) in enumerate(val_loader):
        test_w = test_w.to(device)
        test_d = test_d.to(device)
        test_r = test_r.to(device)
        test_t = test_t.to(device)
        if (test_t[:,:10,:,0]==1).sum() > 0:
            import pdb
            pdb.set_trace()
        output = net([test_w, test_d, test_r],[])
        # pred = scaler_torch.inverse_transform(output)
        # true_value = scaler_torch.inverse_transform(test_t)
        pred = output[0,:,:,0]
        true_value = test_t[0,:,:,0]
