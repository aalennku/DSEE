# from Bio import SeqIO
# import Levenshtein as L
from tqdm import tqdm

import os
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.utils import shuffle
import time
import math

import sys
import argparse
from torch.nn import MSELoss, L1Loss

from model_shipyard.model_shipyard import M_convED_10, M_convED_5, M_GRU, M_RNN

from sklearn.metrics import accuracy_score

np.random.seed(0)

class Twin(torch.nn.Module):
    def __init__(self, model, metric='SE', rescale=True):
        super(Twin, self).__init__()
        self.model = model
        self.rescale_dict = dict()
        self.rescale_dict['SE'] = 1/1.4142135623730951
        self.rescale_dict['L1'] = 1/1.1283791670955126
        self.rescale_dict['EU'] = 6.344349953316304
        assert metric in self.rescale_dict
        self.metric = metric
        self.rescale = rescale
        
    def forward(self, x):
        
        x, y = torch.unbind(x, dim=1)
        if self.rescale:
            xx = self.model(x) * self.rescale_dict[self.metric]
            yy = self.model(y) * self.rescale_dict[self.metric]
        if self.metric == 'SE':
            return torch.sum((xx-yy)**2,dim=-1)
        elif self.metric == 'L1':
            return torch.sum(torch.abs(xx-yy),dim=-1)
        elif self.metric == 'EU':
            return torch.linalg.norm(xx-yy,dim=-1)

def load_data(path='dummy_data.npz'):
    print('loading training data...')
    with np.load(path) as f:
        data_unbatch = np.array(f['data_unbatch'],dtype=np.float32)
        y_unbatch = np.array(f['y_unbatch'],dtype=np.float32)
    print(y_unbatch.shape)
    data_unbatch = data_unbatch[y_unbatch != 0]
    y_unbatch = y_unbatch[y_unbatch != 0]
    print(y_unbatch.shape)
    return data_unbatch, y_unbatch

def main(seed, args):
    np.random.seed(seed)
    args_dict = vars(args)

    models = {'M_convED_5': M_convED_5, 
          'M_convED_10': M_convED_10,
          'M_RNN': M_RNN,
          'M_GRU': M_GRU,}
    train_data_unbatch, y_unbatch = load_data('dummy_data.npz')
    train_data_unbatch, y_unbatch = shuffle(train_data_unbatch, y_unbatch)
    train_data = DataLoader(list(zip(train_data_unbatch, y_unbatch)), batch_size=args.batch_size)
    test_data_unbatch, test_y_unbatch = load_data('dummy_data.npz')
    test_data = DataLoader(list(zip(test_data_unbatch,test_y_unbatch)), batch_size=args.batch_size, shuffle=True)

    
    losses = {'l1loss': L1Loss(),
              'mse': MSELoss(),
              'chi2': chi_sqare_loss,}
    
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    model_save_path = os.path.join(args.save_path, args.model)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_save_path = os.path.join(model_save_path,'{}-{}'.format(seed, time.strftime('%Y-%m-%d', time.localtime())))
    
    for key in np.sort(list(args_dict.keys())):
        if key == 'save_path':
            continue
        model_save_path += '-{}-{}'.format(key,args_dict[key])
        
    print(model_save_path)
    m = models[args.model](output_dim=args.output_dim)
    
    tmodel = Twin(m,args.metric,args.rescale)
    tmodel.to(device)
    
    optimizer = torch.optim.Adam(tmodel.parameters(), lr=args.lr)
    loss = losses[args.loss]
    
    loss_recorder_list = []
    for epoch in range(args.epochs):
        loss_value, loss_recorder = train(tmodel, train_data, epoch, optimizer, total_epochs=args.epochs, loss=loss)
        print('Loss: {}'.format(loss_value))
        loss_recorder_list.append(loss_recorder)
        if (epoch+1) % 5 == 0:
            torch.save(m.state_dict(), model_save_path+'_epoch{}_model.pth'.format(epoch+1))    
        
    torch.save(m.state_dict(), model_save_path+'_model.pth')
    np.save(model_save_path+'_loss_recorder.npy',loss_recorder_list)
    
    y_true, y_pred = test(tmodel, test_data)
    np.savez(model_save_path+'_y_true_y_pred.npz', y_true=y_true, y_pred=y_pred)
    est_error = np.mean(np.abs(y_true-y_pred))
    est_error_relative = np.mean(np.abs(y_true-y_pred)*(y_true!=0)/(y_true+1e-10))
    K = 40
    accuracy = accuracy_score(y_true>K, y_pred>K)
    
    print('\ntest loss: {}'.format(est_error))
    print('test loss_percent: {}'.format(est_error_relative))
    print('test accuracy: {}'.format(accuracy))
    with open(model_save_path+'_test_results.txt','w') as f:
        f.write('test est_error: {}\n'.format(est_error))
        f.write('test est_error_percent: {}\n'.format(est_error_relative))
        f.write('test accuracy: {}\n'.format(accuracy))


def test(tmodel, test_data):
    tmodel.eval()
    y_true = []
    y_pred = []
    for data, y in tqdm(test_data):
        y_true.append(y.numpy())
        output = tmodel(data.to(device))
        y_pred.append(output.cpu().detach().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

def chi_sqare_loss(output,target):
    neg_log = -(-target/2-1.442695*torch.lgamma(target/2)-output/2*1.442695 + (target/2-1)*torch.log2(output))
    return torch.mean(neg_log)


def train(tmodel, train_data, epoch, optimizer, total_epochs, loss):
    tmodel.train()

    loss_all = 0.
    running_loss = 0.
    t_start = time.time()
    loss_recorder = []
    for idx,(data,y) in enumerate(train_data):
        y = y.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = tmodel(data)

        loss_value = loss(output,y)
        
        loss_value.backward()
        running_loss += loss_value.item()
        loss_recorder.append(loss_value.item())
        loss_all += loss_value.item()
        optimizer.step()
        if idx % 5 == 1:
            sys.stdout.write('\r[%d, %5d, %d] loss: %.5f' %
                  (epoch + 1, idx + 1, len(train_data), running_loss / 5))
            sys.stdout.flush()
            running_loss = 0.0
    print('\ntime: {}'.format(time.time()-t_start))
    if total_epochs != 1:
        if (epoch+1) % (total_epochs//2) == 0:
            for param_group in optimizer.param_groups:
                print('changing lr from {} to {}.'.format(param_group['lr'], 0.1 * param_group['lr']))
                param_group['lr'] = 0.1 * param_group['lr']
    return loss_all/len(train_data), loss_recorder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--model', type=str, default='M_convED_5', help='models: M_convED_10, M_convED_5, M_GRU, M_RNN')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--output_dim', type=int, default=80, help='output_dim')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--save-path',type=str, default='./results')
    parser.add_argument('--metric',type=str, default='SE',help='SE: squared Euclidean, EU: Euclidean, L1')
    parser.add_argument('--rescale',type=bool, default=True)
    parser.add_argument('--loss',type=str, default='chi2', help='l1loss, mse, chi2')

    args = parser.parse_args()
    
    for seed in range(5):
        main(seed,args)