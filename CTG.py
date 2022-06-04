import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import tqdm
from Datasets import Datasets

dataset = Datasets("C:\Attention\data\\train.txt")

dataset.bulid_vocab(dataset.en_data,dataset.ch_data)

dataloader = DataLoader(dataset, batch_size=16, num_workers=0,collate_fn=dataset.collate_fn)


maxlen = 128
d_model = 512
units = 512
dropout_rate = 0.2
numofblock = 4
numofhead = 4
# encoder_vocab = len(dataset.ch_vocab)
vocab_size = len(dataset.en_vocab)
epochs = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_padding_mask(seq_q,seq_k):
    # print(seq_k.shape)
    # print(seq_q.shape)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    padding_mask = seq_k.data.eq(1).unsqueeze(1)
    return padding_mask.expand(batch_size,len_q,len_k)


class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size)

    def forward(self,x):
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
        a = torch.matmul(q_,k_.permute(0,1,3,2)) # a = q * k^T(后两个维度)
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
        a = self.dropout(a)
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
        super(EncoderLayer, self).__init__()
        self.mask_self_attention = MultiHeadAttention(units,numofhead,dropout_rate,True)
        self.fc = FC(d_model)

    def forward(self,inputs,padding_mask):
        outputs = self.mask_self_attention(inputs,inputs,inputs,padding_mask,None)
        outputs = self.fc(outputs)
        return outputs.to(DEVICE)


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(units,numofhead,dropout_rate,mask=False)
        self.fc = FC(d_model)


    def forward(self,enc_outputs,padding_mask):
        outputs = self.self_attention(enc_outputs,enc_outputs,enc_outputs,None,padding_mask)
        outputs = self.fc(outputs)
        return outputs.to(DEVICE)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(numofblock)])

    def forward(self,x,padding_mask):
        for layer in self.layers:
            x = layer(x,padding_mask)
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super(Encoder, self).__init__()
        self.pe = PositionalEncoding()
        self.embedding = TokenEmbedding(vocab_size,units)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(numofblock)])

    def forward(self,inputs):
        outputs = self.embedding(inputs)
        outputs = self.pe(outputs.transpose(0, 1)).transpose(0, 1)

        padding_mask = get_padding_mask(inputs,inputs)
        for layer in self.layers:
            outputs = layer(outputs,padding_mask)
        return outputs,padding_mask



class CTG(nn.Module):
    def __init__(self,vocab_size):
        super(CTG, self).__init__()
        self.Encoder = Encoder(vocab_size)
        self.Decoder = Decoder()
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x,epoch=None):
        enc_outputs,padding_mask = self.Encoder(x)
        dec_outputs = self.Decoder(enc_outputs,padding_mask)
        logits = self.linear(dec_outputs)

        logits = logits.view(-1, logits.size(-1))
        return logits

model = CTG(vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in tqdm.tqdm(range(epochs)):
    total = []
    for _,dec_inputs,dec_outputs in dataloader:

        dec_inputs,dec_outputs= dec_inputs.to(DEVICE),dec_outputs.to(DEVICE)

        outputs = model(dec_inputs,epoch)

        loss = criterion(outputs,dec_outputs.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        total.append(loss)
        optimizer.step()
    print(sum(total)/len(total))

def greedy_decoder(model, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    inputs = torch.zeros(1, 0).long()
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：inputs序列会一点点变长（每次添加一个新预测出来的单词）
        inputs = torch.cat([inputs.to(DEVICE), torch.tensor([[next_symbol]], dtype=inputs.dtype).to(DEVICE)],
                              -1)
        # print("inputs:")
        # print(inputs)
        dec_outputs,padding_mask = model.Encoder(inputs)
        dec_outputs = model.Decoder(dec_outputs,padding_mask)
        dec_outputs = model.linear(dec_outputs)
        # projected = model.linear(dec_outputs)
        prob = dec_outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
        # print("prob:")
        # print(dataset.idx2enwords(prob))
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_symbol = next_word
        # print(dataset.idx2en(next_word))
        if next_symbol == dataset.en_vocab["<eos>"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [inputs.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = inputs[:, 1:]
    return greedy_dec_predict

for i in range(20):
    greedy_dec_predict = greedy_decoder(model, start_symbol=dataset.en_vocab["<bos>"])
    # print(input[i], '->', greedy_dec_predict.squeeze())
    print(" ".join([dataset.idx2en(n.item()) for n in greedy_dec_predict.squeeze()]))
