from layers import *
import torch.nn.functional as F

class Bert4Rec(nn.Module):

    def __init__(self, n_items, N, 
                hidden_dim:int = 512, 
                num_head:int = 8, 
                inner_dim:int = 2048, 
                max_length:int = 100):
        super().__init__()
        self.n_items = n_items # the number of items
        self.N = N # the number of layers to be repeated..
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(num_embeddings= n_items, embedding_dim= hidden_dim, padding_idx= 0)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)]
        )
        # self.projection = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, n_items)
        # )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        device= input.device
        bs, item_len = input.shape[:2] # 0, 1
        mask = makeMask(input, option = 'padding').to(device)
        pos = torch.arange(0, item_len).unsqueeze(0).repeat(bs, 1).to(device)
        # (bs, item_len)
        # [[0, 1, 2, ..., item_len-1],...,[0, 1, 2, ..., item_len-1]]

        # Embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))

        # Encoder layers
        for enc_layer in self.enc_layers:
            output = enc_layer(output, mask)
        # (bs, item_len, hidden_dim)
        output = output[:, -1, :] # (bs, hidden_dim)
        # output = self.projection(output)
        output = output@(self.embedding.weight.T)
        return output

class GLBert4Rec(nn.Module):

    def __init__(self, n_items, N, 
                hidden_dim:int = 512, 
                num_head:int = 8, 
                inner_dim:int = 2048, 
                max_length:int = 100):
        super().__init__()
        self.n_items = n_items # the number of items
        self.N = N # the number of layers to be repeated..
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(num_embeddings= n_items+1, embedding_dim= hidden_dim, padding_idx= 0) # including padding...
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)]
        )

        self.dropout = nn.Dropout(0.1)

        self.projection_att = nn.Linear(hidden_dim, 1, bias=False)
        # to obtain a global attention
        
    def forward(self, input):
        device= input.device
        bs, item_len = input.shape[:2] # 0, 1
        mask = makeMask(input, option = 'padding').to(device)
        pos = torch.arange(0, item_len).unsqueeze(0).repeat(bs, 1).to(device)
        # (bs, item_len)
        # [[0, 1, 2, ..., item_len-1],...,[0, 1, 2, ..., item_len-1]]

        # Embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))

        # Encoder layers
        for enc_layer in self.enc_layers:
            output = enc_layer(output, mask)
            # (bs, item_len, hidden_dim)

        att = self.projection_att(output)
        # att = att / att.sum(dim= 1, keepdim= True)
        att = F.softmax(att, dim=1)
        # (bs, item_len, 1)
        output = output.permute(0,2,1)
        # (bs, hidden_dim, item_len)
        output = (output @ att).squeeze() 
        # (bs, hidden_dim)
        output = output.unsqueeze(1)
        # (bs, 1, hidden_dim)
        # obtain scores
        output = output@(self.embedding.weight.T)
        # (bs, 1, hidden_dim) @ (hidden_dim, n_items)
        # (bs, 1, V)
        output = output.squeeze()
        # (bs, n_items)

        return output

# bert4rec = GLBert4Rec(10, 2, 10,2,20)
# input = torch.LongTensor(
#     [[1,2,3,4,5,0,0],
#      [6,7,8,0,0,0,0]]
# )
# output = bert4rec(input)
# print(output.shape) # 2,10