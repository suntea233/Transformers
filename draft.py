
```python
import jieba
import re
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Datasets(Dataset):
    def __init__(self,path):
        super(Datasets, self).__init__()
        self.path = path
        self.ch_vocab = None
        self.en_vocab = None
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        self.en_data,self.ch_data = self.preprocessing()
        self.bulid_vocab(self.en_data,self.ch_data)

    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)


    def __getitem__(self, item):
        en = self.en_data[item]
        ch = self.ch_data[item]
        en = self.words2idx(en,'en')
        ch = self.words2idx(ch,'ch')
        return en,ch


    def words2idx(self,words,language):
        res = []
        if language == 'en':
            for word in words:
                res.append(self.en_vocab[word])
        else:
            for word in words:
                res.append(self.ch_vocab[word])
        return torch.tensor(res)


    def preprocessing(self,train=True):
        en = []
        ch = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                sentences = re.split(',| ', line[0].lower())
                en.append(["<bos>"] + list(filter(lambda x: x != '', sentences)) + ["<eos>"])
                # test时tgt不加开头结束，用于BLEU计算
                if train:
                    ch.append(["<bos>"] + jieba.lcut(line[1]) + ["<eos>"])
                else:
                    ch.append(jieba.lcut(line[1]))
        return en,ch



    def yield_tokens(self,data_iter):
        for data_sample in data_iter:
            yield data_sample


    def bulid_vocab(self,en,ch):
        self.en_vocab = build_vocab_from_iterator(self.yield_tokens(en),min_freq=1,specials=self.special_symbols,special_first=True)
        self.ch_vocab = build_vocab_from_iterator(self.yield_tokens(ch),min_freq=1,specials=self.special_symbols,special_first=True)
        self.en_vocab.set_default_index(self.UNK_IDX)
        self.ch_vocab.set_default_index(self.UNK_IDX)


    def idx2ch(self,id):
        return self.ch_vocab.get_itos()[id]

    def idx2en(self,id):
        return self.en_vocab.get_itos()[id]


    def idx2enwords(self,ids):
        return ' '.join([self.idx2en(id) for id in ids])


    def idx2chwords(self,ids):
        return ' '.join([self.idx2ch(id) for id in ids])


    def collate_fn(self, batch_list):
        en_inputs_index, en_outputs_index, ch_index = [], [], []
        enPAD = self.en_vocab['<pad>']
        chPAD = self.ch_vocab['<pad>']

        for en, ch in batch_list:

            en_inputs_index.append(en[:-1])
            en_outputs_index.append(en[1:])
            ch_index.append(ch)

        en_inputs_index = pad_sequence(en_inputs_index, padding_value=enPAD)
        en_outputs_index = pad_sequence(en_outputs_index, padding_value=enPAD)
        ch_index = pad_sequence(ch_index, padding_value=chPAD)
        # if not self.batch_first:
        en_inputs_index = en_inputs_index.transpose(0, 1)
        en_outputs_index = en_outputs_index.transpose(0, 1)
        ch_index = ch_index.transpose(0, 1)

        return ch_index, en_inputs_index, en_outputs_index


```
```python
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
# from Datasets import Datasets
import tqdm
import numpy as np
dataset = Datasets("../input/attention/train.txt")
dataloader = DataLoader(dataset, batch_size=16, num_workers=0,collate_fn=dataset.collate_fn)


maxlen = 128
d_model = 512
units = 512
dropout_rate = 0.2
numofblock = 4
numofhead = 4
encoder_vocab = len(dataset.ch_vocab)
decoder_vocab = len(dataset.en_vocab)
epochs = 25

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_padding_mask(seq_q,seq_k):
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
        a = torch.matmul(q_,k_.permute(0,1,3,2)) # a = q * k^T(后两个维度)
        a = a / (k_.size()[-1] ** 0.5) # shape:(batch_size,num_head,seq_len,seq_len)
        batch_size_shape = a.shape[0]
        seq_len_shape = a.shape[2]
        if self.mask:
            self_padding_mask = self_padding_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            masked = torch.zeros((batch_size_shape,1,seq_len_shape,seq_len_shape))
            masked = Variable((1 - torch.tril(masked, diagonal=0)) * (-2 ** 32 + 1)).to(DEVICE)

            assert masked.shape[-1] == self_padding_mask.shape[-1]
            a = a + masked+ self_padding_mask
        else:
            enc_dec_padding_mask = enc_dec_padding_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

            a = a + enc_dec_padding_mask

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
        outputs = self.layer1(x)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs += x
        outputs = self.LayerNormalization(outputs)
        return outputs.to(DEVICE)



class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = TokenEmbedding(maxlen,units)


    def forward(self,x):
        return self.pe(Variable(torch.unsqueeze(torch.arange(0, x.size()[1]).to(DEVICE), 0).repeat(x.size(0), 1).long())).to(DEVICE)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_units=units,num_heads=4,dropout_rate=dropout_rate)
        self.fc =FC(input_channels=d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self,x,padding_mask):
        attention_score = self.self_attention(x,x,x,self_padding_mask=None,enc_dec_padding_mask=padding_mask)
        outputs = attention_score + x
        outputs = self.ln(outputs)
        outputs = self.fc(outputs)
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

        dec_outputs = dec_outputs + dec_inputs
        dec_outputs = self.ln(dec_outputs)
        dec_outputs = self.fc(dec_outputs)

        dec_enc_outputs = self.enc_dec_attention(dec_outputs,enc_outputs,enc_outputs,self_padding_mask=None,enc_dec_padding_mask=enc_dec_padding_mask)
        dec_enc_outputs = dec_enc_outputs + dec_outputs
        dec_enc_outputs = self.ln(dec_enc_outputs)
        dec_enc_outputs = self.fc(dec_enc_outputs)
        return dec_enc_outputs


class Encoder(nn.Module):
    def __init__(self,encoder_vocab):
        super().__init__()
        self.embedding = TokenEmbedding(encoder_vocab,units)
        self.pe = PositionEncoding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(numofblock)])

    def forward(self,x):
        enc_outputs = self.embedding(x)
        enc_outputs = self.pe(enc_outputs)
        padding_mask = get_padding_mask(x,x)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs,padding_mask)
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self,decoder_vocab):
        super().__init__()
        self.embedding = TokenEmbedding(decoder_vocab,units)
        self.pe = PositionEncoding()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(numofblock)])

    def forward(self,dec_inputs,enc_inputs,enc_outputs):
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


    def forward(self,enc_inputs,dec_inputs):

        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        logits = self.linear(dec_outputs)

        logits = logits.view(-1, logits.size(-1))

        return logits

model = Transformers(encoder_vocab,decoder_vocab).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
#
for epoch in tqdm.tqdm(range(epochs)):
    total = []
    for enc_inputs,dec_inputs,dec_outputs in dataloader:

        enc_inputs,dec_inputs,dec_outputs= enc_inputs.to(DEVICE),dec_inputs.to(DEVICE),dec_outputs.to(DEVICE)

        outputs = model(enc_inputs,dec_inputs)
        # outputs = outputs.max(dim=-1, keepdim=False)[1]
#         print(outputs)

#         print(dec_outputs.contiguous().view(-1))
        loss = criterion(outputs,dec_outputs.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        total.append(loss)
        optimizer.step()
    print(sum(total)/len(total))



def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
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
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(DEVICE), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(DEVICE)],
                              -1)
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.linear(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
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
test_data = Datasets('../input/attention/train.txt')
# test_data = Datasets('C:\Attention\data\\test.txt')
test_loader = DataLoader(test_data, batch_size=16, num_workers=0,collate_fn=dataset.collate_fn)
enc_inputs,dec_inputs,dec_outputs = next(iter(test_loader))


print()
print("="*30)
print(enc_inputs)
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(DEVICE), start_symbol=dataset.en_vocab["<bos>"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([test_data.idx2ch(t.item()) for t in enc_inputs[i]], '->',
          [test_data.idx2en(n.item()) for n in greedy_dec_predict.squeeze()])


```