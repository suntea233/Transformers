import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import tqdm
from model import Encoder,Decoder,encoder_vocab,decoder_vocab
from Datasets import Datasets

d_model = 512
latent_dim = 512
epochs = 1
dataset = Datasets("C:\Attention\data\\train.txt")

dataset.bulid_vocab(dataset.en_data,dataset.ch_data)

dataloader = DataLoader(dataset, batch_size=16, num_workers=0,collate_fn=dataset.collate_fn)



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(d_model,512),
            nn.ReLU(),
            nn.Linear(512,20),
            nn.ReLU(),
        )
        self.mean = nn.Linear(20,1)
        self.log_var = nn.Linear(20,1)

        # self.Decoder = nn.Sequential(
        #     nn.Linear(1,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        # )
        self.fc1 = nn.Linear(1,512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512,512)


    def reparameterize(self,z_mean,z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean

    def forward(self,x):
        # print(x.type())
        x = self.Encoder(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.reparameterize(z_mean,z_log_var)
        # print(z.shape)
        # print(z.type())
        # print(z)
        # outputs = self.Decoder(z)
        outputs = self.fc1(z)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        return z,z_mean,z_log_var,outputs


class Transformer_VAE(nn.Module):
    def __init__(self,encoder_vocab,decoder_vocab):
        super(Transformer_VAE, self).__init__()
        self.encoder = Encoder(encoder_vocab)
        self.vae = VAE().to(DEVICE)
        self.decoder = Decoder(decoder_vocab)
        self.linear = nn.Linear(d_model,decoder_vocab).to(DEVICE)

    def forward(self,enc_inputs,dec_inputs):
        # print(enc_inputs.type())
        vae_inputs = self.encoder(enc_inputs)
        # print(vae_inputs)
        enc_outputs,z_mean,z_log_var,vae_outputs = self.vae(vae_inputs)

        dec_outputs = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        logits = self.linear(dec_outputs)
        logits = logits.view(-1, logits.size(-1))
        return logits,z_mean,z_log_var,vae_inputs,vae_outputs


model = Transformer_VAE(encoder_vocab,decoder_vocab).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
optimizer_vae = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-6)

for epoch in tqdm.tqdm(range(epochs)):
    total = []
    for enc_inputs,dec_inputs,dec_outputs in dataloader:
        enc_inputs,dec_inputs,dec_outputs= enc_inputs.to(DEVICE),dec_inputs.to(DEVICE),dec_outputs.to(DEVICE)
        outputs,z_mean,z_log_var,vae_inputs,vae_outputs = model(enc_inputs,dec_inputs)
        loss1 = criterion(outputs,dec_outputs.contiguous().view(-1))


        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + z_mean ** 2 - 1. - z_log_var, 1))
        print(vae_inputs.shape)
        print(vae_outputs.shape)
        reconstruction_loss = F.mse_loss(vae_inputs,vae_outputs)


        loss = loss1+kl_loss+reconstruction_loss
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

        dec_inputs = dec_input.float()
        print(dec_inputs.shape)
        dec_inputs = model.vae(dec_inputs)
        dec_inputs = dec_inputs.long()
        dec_outputs = model.decoder(dec_inputs, enc_input, enc_outputs)
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
