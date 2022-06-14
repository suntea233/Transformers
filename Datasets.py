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
import os
import nltk
import pandas as pd


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


    def build_vocab(self,en,ch):
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







class IMDB(Dataset):
    def __init__(self,path):
        super(IMDB, self).__init__()
        self.vocab = None
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.path = path

        # self.generate_csv()
        self.csv = True

        self.data,self.labels = self.preprocessing(path,self.csv)

        self.build_vocab(self.data)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        words = self.data[item]
        labels = self.labels[item]
        # print(type(words))
        return self.words2idx(words),labels


    def words2idx(self,words):
        res = []
        for word in words:
            res.append(self.vocab[word])
        return torch.tensor(res)


    def preprocessing(self,url,csv=False):
        if csv:
            temp = pd.read_csv("C:\Attention\\aclImdb\\train\\imdb.csv")
            sentences= []
            for sentence in temp['sentences']:
                sentences.append(eval(sentence))
            labels = temp['labels'].tolist()
            return sentences,labels
        else:
            file = os.listdir(url)
            sentences = []
            for f in file:
                real_url = os.path.join(url,f)
                with open(real_url,'r',encoding='utf-8') as fread:
                    temp = fread.read().replace("<br />","").replace("(","").replace(")","").replace(".","").replace(',',"").replace("``","").replace("'","").replace('"',"").replace("-","").replace("#","").lower()
                    temp = nltk.word_tokenize(temp)
                    sentences.append(["<bos>"]+temp+['<eos>'])
            return sentences

    def idx2word(self,id):
        return self.vocab.get_itos()[id]


    def idx2words(self,ids):
        return ' '.join([self.idx2word(id) for id in ids])


    def generate_csv(self):
        neg_sentences = self.preprocessing("C:\Attention\\aclImdb\\train\\neg")
        print("neg completely")

        pos_sentences = self.preprocessing("C:\Attention\\aclImdb\\train\\pos")
        print("pos completely")
        neg_labels = [0 for _ in range(len(neg_sentences))]
        pos_labels = [1 for _ in range(len(pos_sentences))]
        result = pd.DataFrame({"sentences": neg_sentences+pos_sentences, "labels": neg_labels+pos_labels})
        result.to_csv("C:\Attention\\aclImdb\\train\\imdb.csv")



    def yield_tokens(self,data_iter):
        for data_sample in data_iter:
            yield data_sample


    def build_vocab(self,word):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(word),min_freq=1,specials=self.special_symbols,special_first=True)
        self.vocab.set_default_index(self.UNK_IDX)


    def collate_fn(self, batch_list):
        inputs_index, outputs_index, labels = [], [], []
        PAD = self.vocab['<pad>']
        for word,label in batch_list:
            inputs_index.append(word[:-1])
            outputs_index.append(word[1:])
            labels.append(label)
        inputs_index = pad_sequence(inputs_index, padding_value=PAD)
        outputs_index = pad_sequence(outputs_index, padding_value=PAD)

        inputs_index = inputs_index.transpose(0, 1)
        outputs_index = outputs_index.transpose(0, 1)

        return inputs_index, outputs_index, labels



# data = IMDB("C:\Attention\\aclImdb\\train\\neg")
# print(data[0])
# dataset = Datasets("C:\Attention\data\\train.txt")
# dataset.build_vocab(dataset.en_data,dataset.ch_data)
# print(dataset[0])
class SST2(Dataset):
    def __init__(self,path):
        super(SST2, self).__init__()
        self.vocab = None
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.path = path


        self.data,self.labels = self.preprocessing(path)
        # print(self.data[-1])
        # for i in self.data:
        #     print(i)
        self.build_vocab(self.data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        words = self.data[item]
        labels = self.labels[item]
        return self.words2idx(words),labels


    def words2idx(self,words):
        res = []
        for word in words:
            res.append(self.vocab[word])
        return torch.tensor(res)


    def preprocessing(self,data_path):
        # 本任务中暂时只用train、test做划分，不包含dev验证集，
        # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
        # 本函数只适用于读取指定文件，不具通用性，仅作示范
        sentences = []
        # categories用于统计分类标签的总数，用set结构去重
        labels = []
        with open(data_path, 'r', encoding="utf8") as file:
            for sample in file.readlines():
                # polar指情感的类别，当前只有两种：
                #   ——0：positive
                #   ——1：negative
                # sent指对应的句子
                polar, sent = sample.strip().split("\t")
                sent = sent.replace("<br />", "").replace("(", "").replace(")", "").replace(".", "").replace(
                    ',', "").replace("``", "").replace("'", "").replace('"', "").replace("-", "").replace("#",
                                                                                                          "").lower()
                sent = nltk.word_tokenize(sent)
                sentences.append(["<bos>"] + sent + ['<eos>'])
                labels.append(int(polar))
        return sentences,labels

    def idx2word(self,id):
        return self.vocab.get_itos()[id]


    def idx2words(self,ids):
        return ' '.join([self.idx2word(id) for id in ids])


    def yield_tokens(self,data_iter):
        for data_sample in data_iter:
            yield data_sample


    def build_vocab(self,word):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(word),min_freq=1,specials=self.special_symbols,special_first=True)
        self.vocab.set_default_index(self.UNK_IDX)


    def collate_fn(self, batch_list):
        inputs_index, outputs_index, labels = [], [], []
        PAD = self.vocab['<pad>']
        for word,label in batch_list:
            inputs_index.append(word[:-1])
            outputs_index.append(word[1:])
            labels.append(label)
        inputs_index = pad_sequence(inputs_index, padding_value=PAD)
        outputs_index = pad_sequence(outputs_index, padding_value=PAD)

        inputs_index = inputs_index.transpose(0, 1)
        outputs_index = outputs_index.transpose(0, 1)

        return inputs_index, outputs_index, labels


data = SST2("C:\Attention\pytorch-master\\bert-sst2\sst2_shuffled.tsv")
print(data.words2idx(""))