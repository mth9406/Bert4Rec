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
# from metric import *
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
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')

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
        model = Bert4Rec(n_items, args.N, args.hidden_dim, args.num_head, args.inner_dim, args.max_length).to(device)
    else: 
        raise Exception("Unknown model!")
    
    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        criterion = nn.CrossEntropyLoss()
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr, val_loss = validate(test_loader, model, criterion)
        print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f},  Val_loss: {:.4f}".format(args.topk, recall, args.topk, mrr, val_loss))
        return    

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    
    early_stopping = EarlyStopping(
            patience= args.patience,
            verbose= True
        )

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)
        scheduler.step()

        recall, mrr, val_loss = validate(valid_loader, model, criterion)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, Val_loss: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr, val_loss))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(val_loss, model, epoch, optimizer)
        # torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')    

def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices) # Bxk
    hits = (targets == indices).nonzero() # B
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()


def get_recall_mrr(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr

def validate(valid_loader, model, criterion):
    model.eval()
    recalls = []
    mrrs = []
    losses = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq)
            loss = criterion(outputs, target)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = get_recall_mrr(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
            losses.append(loss.item())
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_val_loss = np.mean(losses)
    return mean_recall, mean_mrr, mean_val_loss

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