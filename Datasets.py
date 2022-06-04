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
    def __init__(self,path,train=True):
        super(Datasets, self).__init__()
        self.path = path
        self.ch_vocab = None
        self.en_vocab = None
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.train = train
        self.en_data,self.ch_data = self.preprocessing()

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
                # print(self.en_vocab[word])
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



# print(data.en_vocab.get_itos()[0])
# print(data.en_vocab.get_itos()[1])
# print(data.en_vocab.get_itos()[2])
# print(data.en_vocab.get_itos()[3])
# print(data.en_vocab.get_itos()[4])
# print(data.en_vocab.get_itos()[5])
# print(data.en_vocab.get_itos()[6])
# print(data.en_vocab.get_itos()[7])