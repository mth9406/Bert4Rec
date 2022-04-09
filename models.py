from layers import *

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
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, n_items)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        bs, item_len = input.shape[:2] # 0, 1
        mask = makeMask(input, option = 'padding')
        pos = torch.arange(0, item_len).unsqueeze(0).repeat(bs, 1) 
        # (bs, item_len)
        # [[0, 1, 2, ..., item_len-1],...,[0, 1, 2, ..., item_len-1]]

        # Embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))

        # Encoder layers
        for enc_layer in self.enc_layers:
            output = enc_layer(output, mask)
        # (bs, item_len, hidden_dim)
        output = output[:, -1, :] # (bs, hidden_dim)
        output = self.projection(output)
        return output

# bert4rec = Bert4Rec(10, 2, 10,2,20)
# input = torch.LongTensor(
#     [[1,2,3,4,5,0,0],
#      [6,7,8,0,0,0,0]]
# )
# output = bert4rec(input)
# print(output.shape) # 2,10