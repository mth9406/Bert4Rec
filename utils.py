import torch
import numpy as np

def collate_fn(data):
    # data
    # list type
    # [[sees_items, target_item]]
    # ex: [ [[1,2,3], 4], [[1,2], 3], ... ]
    # ->  [ [[1,2,3], 4], [[1,2, 0],3], ...]
    data.sort(key= lambda x: len(x[0]), reverse= True)
    # sort the data in a descending item length order
    lens = [len(sess) for sess, label in data] # seesion lengths
    # ex: [3, 2, ...]
    labels = []
    padded_sess = torch.zeros(len(data), max(lens)).long() # (nobs, max(lens))
    for i, (sess, label) in enumerate(data):
        padded_sess[i, :lens[i]] = torch.LongTensor(sess)
        # [1,2,3] -> [1,2,3,0,..0] padded 0 up to a predefined max len.
        labels.append(label)
    # padded_sess = padded_sess.transpose(0,1) 
    return padded_sess, torch.tensor(labels).long(), lens

class EarlyStopping(object):

    def __init__(self, 
                patience: int= 10, 
                verbose: bool= False, delta: float= 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta # significant change

        self.best_score = None
        self.early_stop= False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0


    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar') 
        self.val_loss_min = val_loss