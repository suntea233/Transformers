from Datasets import IMDB
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import tqdm
data = IMDB("C:\Attention\\aclImdb\\train\\neg")

dataloader = DataLoader(data, batch_size=16, num_workers=0,collate_fn=data.collate_fn,shuffle=True)

print("data completely")
maxlen = 128
d_model = 512
units = 512
dropout_rate = 0.2
numofblock = 4
numofhead = 2
# encoder_vocab = len(dataset.ch_vocab)
vocab_size = len(data.vocab)
epochs = 10
latent_dim = 512
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
        # print(x.shape)

        return self.embedding(x).to(DEVICE) # shape = (batch_size,input_seq_len,emb_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, units, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, units)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, units, 2).float() * (-math.log(10000.0) / units))
        # print(position.shape)
        # print(div_term.shape)
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
    def __init__(self,input_channels,units=(2048,d_model)):
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


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mask_self_attention = MultiHeadAttention(d_model,numofhead,dropout_rate,True)
        self.fc = FC(d_model)


    def forward(self,inputs,padding_mask):
        outputs = self.mask_self_attention(inputs,inputs,inputs,padding_mask,None)
        outputs = self.fc(outputs)
        return outputs.to(DEVICE)



class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model,numofhead,dropout_rate,mask=False)
        self.fc = FC(d_model)



    def forward(self,enc_outputs):
        outputs = self.fc(enc_outputs)
        return outputs.to(DEVICE)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(numofblock)])
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(512, 256)

    def forward(self,x,labels):
        x,y = self.fc2(x),self.fc1(labels)
        x = torch.cat((x,y),dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super(Encoder, self).__init__()
        self.pe = PositionalEncoding(units)
        self.embedding = TokenEmbedding(vocab_size,units)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(numofblock)])
        self.fc1 = nn.Linear(2,256)
        self.fc2 = nn.Linear(512,256)


    def forward(self,inputs,labels):
        outputs = self.embedding(inputs)
        outputs = self.pe(outputs.transpose(0, 1)).transpose(0, 1)
        seq_len = inputs.size()[1]
        labels = F.one_hot(labels,num_classes=2)
        labels = labels.unsqueeze(1)
        labels = labels.repeat(1, seq_len, 1).float()
        # print(self.fc1(labels))
        x = self.fc2(outputs)
        y = self.fc1(labels)
        outputs = torch.cat((x,y),dim=-1)
        print(outputs.shape)
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
        self.mean = nn.Linear(d_model,latent_dim)
        self.log_var = nn.Linear(d_model,latent_dim)


    def reparameterize(self,z_mean,z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean


    def forward(self,inputs,labels):

        enc_outputs,padding_mask = self.Encoder(inputs,labels)

        z_mean = self.mean(enc_outputs)
        z_log_var = self.log_var(enc_outputs)

        z = self.reparameterize(z_mean,z_log_var)
        enc_outputs = self.Decoder(z,labels)
        logits = self.linear(enc_outputs)

        logits = logits.view(-1, logits.size(-1))
        return logits,z_mean,z_log_var



model = CTG(vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


for epoch in tqdm.tqdm(range(epochs)):
    total = []
    for inputs,outputs,labels in dataloader:
        labels = torch.tensor(labels)
        inputs,outputs,labels= inputs.to(DEVICE), outputs.to(DEVICE), labels.to(DEVICE)

        logits,z_mean,z_log_var = model(inputs,labels)

        normal_loss = criterion(logits,outputs.contiguous().view(-1))

        reconstruction_loss = F.cross_entropy(logits,outputs.contiguous().view(-1))
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean ** 2 - 1. - z_log_var, 1))

        loss = normal_loss + reconstruction_loss + kl_loss
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
        if next_symbol == data.vocab["<eos>"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [inputs.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = inputs[:, 1:]
    return greedy_dec_predict

for i in range(20):
    greedy_dec_predict = greedy_decoder(model, start_symbol=data.vocab["<bos>"])
    # print(input[i], '->', greedy_dec_predict.squeeze())
    print(" ".join([data.idx2word(n.item()) for n in greedy_dec_predict.squeeze()]))
