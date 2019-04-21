import torch
import torch.utils.data as data_util
import numpy as np


class Datasets():
    def __init__(self, config):
        self.train_src = self._get_datasets(config.filename_train_src)
        self.train_tgt = self._get_datasets(config.filename_train_tgt)
        self.valid_src = self._get_datasets(config.filename_valid_src)
        self.valid_tgt = self._get_datasets(config.filename_valid_tgt)
        self.test_src = self._get_datasets(config.filename_test_src)
        self.test_tgt = self._get_datasets(config.filename_test_tgt)

    def _get_datasets(self, filename):
        result = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # print(line)
                result.append(line)
        return result


# save pt
def get_trimmed_datasets_src(datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        line = line.strip().split(' ')
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['[SEP]']
                # print(sen)
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['[UNK]']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        k += 1
    data = torch.from_numpy(data).type(torch.LongTensor)
    return data


def get_trimmed_datasets_tgt(datasets, tokenizer, max_length, eos):
    data_ids = np.zeros([len(datasets), max_length])
    for i, line in enumerate(datasets):
        # word to index
        line = tokenizer.tokenize(line)
        line = tokenizer.convert_tokens_to_ids(line)
        line.append(eos)
        if len(line) <= max_length:
            line = np.pad(np.array(line), (0, max_length-len(line)), 'constant')
        else:
            line = line[:max_length]
        data_ids[i] = line

    data_ids = torch.from_numpy(data_ids).type(torch.LongTensor)
    return data_ids


def save_data(text, summary, word2idx, tokenizer, t_len, s_len, filename):
    text = get_trimmed_datasets_src(text, word2idx, t_len)
    summary = get_trimmed_datasets_tgt(summary, tokenizer, s_len, 102)
    data = data_util.TensorDataset(text, summary)
    print('data save at ', filename)
    torch.save(data, filename)


# convert idx to words, if idx <bos> is stop, return sentence
def index2sentence(index, idx2word):
    sen = []
    for i in range(len(index)):
        if idx2word[index[i]] == '[SEP]':
            break
        if idx2word[index[i]] == '<S>':
            continue
        else:
            sen.append(idx2word[index[i]])
    if len(sen) == 0:
        sen.append('[UNK]')
    return sen


def data_load(filename, batch_size, shuffle):
    data = torch.load(filename)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=2)
    return data_loader