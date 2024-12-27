import numpy as np
from lib.utils_model_pre1 import *

num_for_predict = 1
num_of_days = 1
num_of_hours = 3
num_of_5min = 12
merge = False

Metro_edge_matrix = np.load('./data/all_5min_50_100.npy')[:,:,:]# 8760，69，69
# Metro_week_matrix = np.load('./data/ext2018_week_Matrix.npy')  # [(45,56),[]]
# Metro_hour_matrix = np.load('./data/ext2018_hour_Matrix.npy')  #
Metro_week_matrix = None
Metro_hour_matrix = None

all_data,scaler = read_and_generate_dataset(Metro_edge_matrix,Metro_week_matrix,Metro_hour_matrix,
                                         num_of_days,
                                         num_of_hours,
                                         num_of_5min,
                                         num_for_predict,
                                         merge)


np.save(r'data/train_day.npy', all_data['train']['day'])
np.save(r'data/train_hour.npy', all_data['train']['hour'])
np.save(r'data/train_recent.npy', all_data['train']['recent'])
np.save(r'data/train_target.npy', all_data['train']['target'])
np.save(r'data/val_day.npy', all_data['val']['day'])
np.save(r'data/val_hour.npy', all_data['val']['hour'])
np.save(r'data/val_recent.npy', all_data['val']['recent'])
np.save(r'data/val_target.npy', all_data['val']['target'])
np.save(r'data/test_day.npy', all_data['test']['day'])
np.save(r'data/test_hour.npy', all_data['test']['hour'])
np.save(r'data/test_recent.npy', all_data['test']['recent'])
np.save(r'data/test_target.npy', all_data['test']['target'])

f=open('data/scaler.txt', 'w')
s = {'mean': scaler.mean, 'std': scaler.std}
f.write(str(s))
f.close()

print(all_data['train']['day'].shape)
print(all_data['train']['hour'].shape)
print(all_data['train']['recent'].shape)
print(all_data['train']['target'].shape)
