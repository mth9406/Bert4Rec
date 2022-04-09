import torch

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
    