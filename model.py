import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from Datasets import Datasets
import tqdm
import numpy as np

dataset = Datasets("C:\Attention\data\\train.txt")

dataset.build_vocab(dataset.en_data,dataset.ch_data)

dataloader = DataLoader(dataset, batch_size=16, num_workers=0,collate_fn=dataset.collate_fn)


maxlen = 128
d_model = 512
units = 512
dropout_rate = 0.2
numofblock = 4
numofhead = 4
encoder_vocab = len(dataset.ch_vocab)
decoder_vocab = len(dataset.en_vocab)
epochs = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_padding_mask(seq_q,seq_k):
    # print(seq_k.size())
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    padding_mask = seq_k.data.eq(1).unsqueeze(1)
    return padding_mask.expand(batch_size,len_q,len_k)


class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size)

    def forward(self,x):
        # print(x.shape)

        return self.embedding(x).to(DEVICE) # shape = (batch_size,input_seq_len,emb_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads, dropout_rate, mask=False):
        super().__init__()
        self.num_units = num_units
        self.num_head = num_heads
        self.dropout_rate = dropout_rate
        self.mask = mask
        self.linearQ = nn.Linear(self.num_units,self.num_units)
        self.linearK = nn.Linear(self.num_units,self.num_units)
        self.linearV = nn.Linear(self.num_units,self.num_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNormalization = nn.LayerNorm(d_model)
        self.Q = nn.Sequential(self.linearQ,self.relu)
        self.K = nn.Sequential(self.linearK,self.relu)
        self.V = nn.Sequential(self.linearV,self.relu)


    def forward(self, queries, keys, values, self_padding_mask, enc_dec_padding_mask):
        '''
        :param queries: shape:(batch_size,input_seq_len,d_model)
        :param keys: shape:(batch_size,input_seq_len,d_model)
        :param values: shape:(batch_size,input_seq_len,d_model)
        :return: None
        '''
        q, k, v = self.Q(queries), self.K(keys), self.V(values)

        q_split, k_split, v_split = torch.chunk(q,self.num_head,dim=-1), torch.chunk(k,self.num_head,dim=-1), torch.chunk(v,self.num_head,dim=-1)
        q_, k_, v_ = torch.stack(q_split,dim=1), torch.stack(k_split,dim=1), torch.stack(v_split,dim=1)
        # shape : (batch_size, num_head, input_seq_len, depth = d_model/num_head)
        a = torch.matmul(q_,k_.permute(0,1,3,2)) # a = q * k^T(???????????????)
        a = a / (k_.size()[-1] ** 0.5) # shape:(batch_size,num_head,seq_len,seq_len)
        batch_size_shape = a.shape[0]
        seq_len_shape = a.shape[2]
        if self.mask:
            self_padding_mask = self_padding_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            masked = torch.ones((batch_size_shape,1,seq_len_shape,seq_len_shape))
            masked = Variable((1 - torch.tril(masked, diagonal=0)) * (-2 ** 32 + 1)).to(DEVICE)

            assert masked.shape[-1] == self_padding_mask.shape[-1]
            a = a + masked
            a.masked_fill_(self_padding_mask,-1e9)
        else:
            enc_dec_padding_mask = enc_dec_padding_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

            a.masked_fill_(enc_dec_padding_mask,-1e9)

        a = F.softmax(a,dim=-1)

        a = torch.matmul(a,v_)
        a = torch.reshape(a.permute(0, 2, 1, 3), shape=(q.shape[0],q.shape[1],q.shape[2]))
        a += queries
        a = self.LayerNormalization(a).to(DEVICE)
        return a


class FC(nn.Module):
    def __init__(self,input_channels,units=(2048,512)):
        super().__init__()
        self.input_channels = input_channels
        self.units = units
        self.layer1 = nn.Linear(self.input_channels,units[0])
        self.layer2 = nn.Linear(self.units[0],self.units[1])
        self.relu = nn.ReLU()
        self.LayerNormalization = nn.LayerNorm(d_model)


    def forward(self,x):
        # print(x.shape)
        outputs = self.layer1(x)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs += x
        outputs = self.LayerNormalization(outputs)
        return outputs.to(DEVICE)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_units=units,num_heads=4,dropout_rate=dropout_rate)
        self.fc =FC(input_channels=d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self,x,padding_mask):
        attention_score = self.self_attention(x,x,x,self_padding_mask=None,enc_dec_padding_mask=padding_mask)
        # outputs = attention_score + x
        outputs = self.fc(attention_score) # shape = (batch_size,seq_len,d_model)
        return outputs.to(DEVICE)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention(units,numofhead,dropout_rate,mask=True)
        self.enc_dec_attention = MultiHeadAttention(units,numofhead,dropout_rate)
        self.fc = FC(d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self,dec_inputs,enc_outputs,self_padding_mask,enc_dec_padding_mask):
        dec_outputs = self.self_attention(dec_inputs,dec_inputs,dec_inputs,self_padding_mask,enc_dec_padding_mask=None)

        # dec_outputs = dec_outputs + dec_inputs
        # dec_outputs = self.ln(dec_outputs)

        dec_outputs = self.fc(dec_outputs)

        dec_enc_outputs = self.enc_dec_attention(dec_outputs,enc_outputs,enc_outputs,self_padding_mask=None,enc_dec_padding_mask=enc_dec_padding_mask)
        # dec_enc_outputs = dec_enc_outputs + dec_outputs
        # dec_enc_outputs = self.ln(dec_enc_outputs)
        dec_enc_outputs = self.fc(dec_enc_outputs)
        return dec_enc_outputs


class Encoder(nn.Module):
    def __init__(self,encoder_vocab):
        super().__init__()
        self.embedding = TokenEmbedding(encoder_vocab,units)
        self.pe = PositionalEncoding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(numofblock)])

    def forward(self,x):
        enc_outputs = self.embedding(x)
        enc_outputs = self.pe(enc_outputs.transpose(0,1)).transpose(0,1)

        padding_mask = get_padding_mask(x,x)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs,padding_mask)
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self,decoder_vocab):
        super().__init__()
        self.embedding = TokenEmbedding(decoder_vocab,units)
        self.pe = PositionalEncoding()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(numofblock)])

    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        # print(dec_inputs.shape)
        # print(enc_outputs)
        dec_outputs = self.embedding(dec_inputs)
        dec_outputs = self.pe(dec_outputs)
        self_padding_mask = get_padding_mask(dec_inputs,dec_inputs).to(DEVICE)
        enc_dec_padding_mask = get_padding_mask(dec_inputs,enc_inputs).to(DEVICE)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs,enc_outputs,self_padding_mask,enc_dec_padding_mask)
        return dec_outputs


class Transformers(nn.Module):
    def __init__(self,encoder_vocab,decoder_vocab):
        super().__init__()
        self.encoder = Encoder(encoder_vocab).to(DEVICE)
        self.decoder = Decoder(decoder_vocab).to(DEVICE)
        self.linear = nn.Linear(d_model,decoder_vocab).to(DEVICE)


    def forward(self,enc_inputs,dec_inputs,epoch):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        logits = self.linear(dec_outputs)
        if epoch==9:
            print(logits)
        logits = logits.view(-1, logits.size(-1))
        return logits

model = Transformers(encoder_vocab,decoder_vocab).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
# #
# for epoch in tqdm.tqdm(range(epochs)):
#     total = []
#     for enc_inputs,dec_inputs,dec_outputs in dataloader:
#
#         enc_inputs,dec_inputs,dec_outputs= enc_inputs.to(DEVICE),dec_inputs.to(DEVICE),dec_outputs.to(DEVICE)
#
#         outputs = model(enc_inputs,dec_inputs,epoch)
#         # print(outputs.shape)
#         # print(dec_outputs.shape)
#         loss = criterion(outputs,dec_outputs.contiguous().view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         total.append(loss)
#         optimizer.step()
#     print(sum(total)/len(total))


#
def greedy_decoder(model, enc_input, start_symbol):
    """????????????
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # ???????????????dec_input????????????????????????????????????????????????????????????????????????
        dec_input = torch.cat([dec_input.to(DEVICE), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(DEVICE)],
                              -1)
        print("inputs:")
        print(dec_input)
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.linear(dec_outputs)

        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # ??????????????????????????????????????????????????????????????????
        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        print("prob:")
        print(dataset.idx2enwords(prob.data))
        next_word = prob.data[-1]  # ???????????????????????????(??????)????????????x'_t???????????????z_t??????????????????????????????????????????z_1,z_2..z_{t-1}
        next_symbol = next_word
        if next_symbol == dataset.en_vocab["<eos>"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict

# enc_inputs, dec_inputs, dec_outputs = dataset.words2idx(sentences,'ch')
test_data = Datasets('C:\Attention\data\\test.txt',False)

test_data.en_vocab = dataset.en_vocab
test_data.ch_vocab = dataset.ch_vocab

# test_data = Datasets('C:\Attention\data\\test.txt')
test_loader = DataLoader(test_data, batch_size=16, num_workers=0,collate_fn = test_data.collate_fn)

enc_inputs,dec_inputs,dec_outputs = next(iter(test_loader))

print()
print("="*30)
# print(enc_inputs)
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(DEVICE), start_symbol=test_data.en_vocab["<bos>"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print(" ".join([dataset.idx2ch(t.item()) for t in enc_inputs[i] if t.item() not in [1,2,3]]), '->',
          " ".join([dataset.idx2en(n.item()) for n in greedy_dec_predict.squeeze()]))
