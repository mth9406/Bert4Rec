import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import load_data, RecSysDataset
from metric import *
from models import Bert4Rec
from utils import *


parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--dataset_path', default='datasets/diginetica/', 
                help='dataset directory path: datasets/diginetica or datasets/yoochoose1_4/ or datasets/yoochoose1_64')

# Training args
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 

# Model configs
parser.add_argument('--model_type', type= str, default= 0, 
                help= "the model type, 0: BERT4Rec")
parser.add_argument('--N', type= int, default= 12, 
                help="the number of transformer encoder layers, default= 12")
parser.add_argument('--hidden_dim', type= int, default= 512,
                help= "the size of hidden dimension, default= 512")
parser.add_argument('--num_head', type= int, default= 8, 
                help= "the number of heads, default= 8")
parser.add_argument('--inner_dim', type= int, default= 2048,
                help= "the size of inner_dim of FCN layer, default= 2048")
parser.add_argument('--max_length', type= int, default= 100,
                help= "max length of a session, default= 100")

# Test, Validation configs
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, 
                help='the number of top score items selected for calculating recall and mrr metrics, default= 20')
parser.add_argument('--valid_portion', type=float, default=0.1,
                help='split the portion of training set as validation set')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Loading data...")
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)

    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    if args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 43098
    elif args.dataset_path.split('/')[-2] in ['yoochoose1_64', 'yoochoose1_4']:
        n_items = 37484
    else:
        raise Exception('Unknown Dataset!')

    if args.model_type == 0:
        model = Bert4Rec(n_items, args.N, args.hidden_dim, args.num_head, args.inner_dim, args.max_length)
    else: 
        raise Exception("Unknown model!")
    
    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(test_loader, model)
        print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
        return    

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)

        recall, mrr = validate(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')    

def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()

if __name__ == '__main__':
    main()
