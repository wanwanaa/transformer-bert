import pickle
import os


class Vocab():
    def __init__(self, config, datasets=None):
        self.filename_idx2word = config.filename_idx2word
        self.filename_word2idx = config.filename_word2idx
        self.vocab_size = config.src_vocab_size
        self.idx2word = []
        self.word2idx = {}

        if datasets is not None:
            self.vocab = self._get_vocab(datasets)
            self.idx2word = self.index2word()
            self.word2idx = self.word2index()
            self.writeFile(self.idx2word, self.filename_idx2word)
            self.writeFile(self.word2idx, self.filename_word2idx)

        else:
            self.idx2word = self.load_vocab(self.filename_idx2word)
            self.word2idx = self.load_vocab(self.filename_word2idx)

    # check whether the given 'filename' exists
    # raise a FileNotFoundError when file not found
    def file_check(self, filename):
        if os.path.isfile(filename) is False:
            raise FileNotFoundError('No such file or directory: {}'.format(filename))

    # load vocab
    def load_vocab(self, filename):
        print('load vocab from', filename)
        self.file_check(filename)
        f = open(filename, 'rb')
        return pickle.load(f)

    # get the vocabulary and sort it by frequency
    def _get_vocab(self, datasets):
        vocab = {}
        for line in datasets:
            line = line.strip().split(' ')
            for c in line:
                flag = vocab.get(c)
                # print(c)
                if flag:
                    vocab[c] += 1
                else:
                    vocab[c] = 0
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        # print(len(vocab))
        return vocab

    # vocab word2idx
    def word2index(self):
        self.word2idx['<pad>'] = 0
        k = 0
        for i in range(self.vocab_size):
            if i == 0:
                self.word2idx['[PAD]'] = i
            elif i == 100:
                self.word2idx['[UNK]'] = i
            elif i == 104:
                self.word2idx['<S>'] = i
            elif i == 102:
                self.word2idx['[SEP]'] = i
            else:
                self.word2idx[self.vocab[k][0]] = i
                k += 1
        return self.word2idx

    # vocab idx2word
    def index2word(self):
        k = 0
        for i in range(self.vocab_size):
            if i == 0:
                self.idx2word.append('[PAD]')
            elif i == 100:
                self.idx2word.append('[UNK]')
            elif i == 104:
                self.idx2word.append('<S>')
            elif i == 102:
                self.idx2word.append('[SEP]')
            else:
                self.idx2word.append(self.vocab[k][0])
                k += 1
        return self.idx2word

    # save vocab in .pkl
    def writeFile(self, vocab, filename):
        with open(filename, 'wb') as f:
            pickle.dump(vocab, f)
        print('vocab saved at:', filename)