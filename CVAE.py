from Datasets import IMDB,SST2
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import tqdm
data = SST2("C:\Attention\pytorch-master\\bert-sst2\sst2_shuffled.tsv")


batch_size = 16
dataloader = DataLoader(data, batch_size=batch_size, num_workers=0,collate_fn=data.collate_fn,shuffle=True)

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
    def __init__(self,input_channels,units=(2048,512)):
        super().__init__()
        self.input_channels = input_channels
        self.units = units
        self.layer1 = nn.Linear(self.input_channels,units[0])
        self.layer2 = nn.Linear(self.units[0],self.units[1])
        # self.layer3 = nn.Linear(514,512)
        self.relu = nn.ReLU()
        self.LayerNormalization = nn.LayerNorm(d_model)


    def forward(self,x,type='D'):
        outputs = self.layer1(x)

        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        # 1. res + linear
        # 2. linear + res
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
        outputs = self.fc(outputs,"E")
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
        self.linear = nn.Linear(514,512)



    def forward(self,x,labels):
        seq_len = x.size()[1]
        labels = F.one_hot(labels,num_classes=2)
        labels = labels.unsqueeze(1)
        labels = labels.repeat(1, seq_len, 1).float()
        labels = labels.float()
        # print(labels.shape)
        # x = self.fc2(x)
        # y = self.fc1(labels)
        x = torch.cat((x,labels),dim=-1)
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super(Encoder, self).__init__()
        self.pe = PositionalEncoding(units)
        self.embedding = TokenEmbedding(vocab_size,units)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(numofblock)])


    def forward(self,inputs):
        outputs = self.embedding(inputs)
        outputs = self.pe(outputs)
        padding_mask = get_padding_mask(inputs,inputs)
        for layer in self.layers:
            outputs = layer(outputs,padding_mask)
        return outputs,padding_mask


class LabelDiscriminator(nn.Module):
    def __init__(self):
        super(LabelDiscriminator, self).__init__()
        self.fc1 = nn.Linear(512+2,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,1024)
        self.fc4 = nn.Linear(1024,512)
        self.fc5 = nn.Linear(512,1)
        self.fc = nn.Sequential(self.fc1,nn.ReLU(),self.fc2,nn.ReLU(),self.fc3,nn.ReLU(),self.fc4,nn.ReLU(),self.fc5)


    def forward(self,x,labels):
        x = x.float()
        seq_len = x.size()[1]

        labels = F.one_hot(labels,num_classes=2)
        labels = labels.unsqueeze(1)
        labels = labels.repeat(1, seq_len, 1).float()
        labels = labels.float()

        x = self.fc(torch.cat((x,labels),dim=-1))
        return x


class CTG(nn.Module):
    def __init__(self,vocab_size):
        super(CTG, self).__init__()
        self.Encoder = Encoder(vocab_size)
        self.Decoder = Decoder()
        self.linear = nn.Linear(d_model,vocab_size)
        self.mean = nn.Linear(d_model,latent_dim)
        self.log_var = nn.Linear(d_model,latent_dim)
        self.D = LabelDiscriminator()
        self.pe = PositionalEncoding(units)
        self.embedding = TokenEmbedding(vocab_size,units)

    def reparameterize(self,z_mean,z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean


    def forward(self,inputs,labels):
        outputs = self.embedding(inputs)
        outputs = self.pe(outputs.transpose(0, 1)).transpose(0, 1)

        enc_outputs,padding_mask = self.Encoder(inputs)

        z_mean = self.mean(enc_outputs)
        z_log_var = self.log_var(enc_outputs)
        z = self.reparameterize(z_mean,z_log_var)
        enc_outputs = self.Decoder(z,labels)
        # print(enc_outputs.shape)
        logits = self.linear(enc_outputs)
        # print(logits.shape)
        logits = logits.view(-1, logits.size(-1))


        real_D = self.D(outputs,labels)
        fake_D = self.D(enc_outputs,labels)



        return logits,z_mean,z_log_var,real_D, fake_D



model = CTG(vocab_size).to(DEVICE)
Discriminator = LabelDiscriminator().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
D_optimizer = optim.AdamW(Discriminator.parameters(),lr=1e-4,weight_decay=1e-6)



for epoch in tqdm.tqdm(range(epochs)):
    total = []
    kl = []
    recon = []
    d = []
    for inputs, outputs, labels in dataloader:
        labels = torch.tensor(labels)
        inputs, outputs, labels = inputs.to(DEVICE), outputs.to(DEVICE), labels.to(DEVICE)

        logits, z_mean, z_log_var, real_D, fake_D = model(inputs, labels)

        d_loss = F.cross_entropy(real_D,torch.ones((batch_size,1)).long().to(DEVICE))
        g_loss = F.cross_entropy(fake_D,torch.zeros((batch_size,1)).long().to(DEVICE))

        normal_loss = criterion(logits, outputs.contiguous().view(-1))
        reconstruction_loss = F.cross_entropy(logits, outputs.contiguous().view(-1))
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean ** 2 - 1. - z_log_var, 1))


        # Discriminator
        D_optimizer.zero_grad()
        d_loss.backward()
        d.append(d_loss)
        D_optimizer.step()

        # transformers
        loss = normal_loss + reconstruction_loss + kl_loss
        optimizer.zero_grad()
        loss.backward()
        total.append(normal_loss)
        kl.append(kl_loss)
        recon.append(reconstruction_loss)
        optimizer.step()

    print("normal_loss = {0},recon_loss = {2},kl_loss = {1}, d_loss = {3}".format(sum(total) / len(total), sum(kl) / len(kl),sum(recon)/len(recon),sum(d)/len(d)))



def get_sequence(dataset):
    s = dataset.words2idx("<bos>".split())
    s = s.unsqueeze(0).to(DEVICE)
    label = torch.tensor([1]).to(DEVICE)
    flag = True
    data = torch.tensor([]).long().to(DEVICE).unsqueeze(0)
    count = 0
    while flag:
        s = torch.cat((s,data),dim=-1)
        dec_outputs,z_mean,z_log_var = model(s.to(DEVICE),label)
        prob = F.softmax(dec_outputs, dim=-1)
        prob = torch.multinomial(prob, num_samples=1)
        data = prob[-1].unsqueeze(0)

        count += 1
        if data == 3:
            flag = False
        if count == 20:
            flag = False

    print(dataset.idx2words(s[-1]))
for i in range(10):
    get_sequence(data)