from utils import *
from pytorch_pretrained_bert import BertTokenizer


def main():
    config = Config()

    print('Loading data ... ...')
    datasets = Datasets(config)
    # print(datasets.test_src)

    # get vocab(idx2word, word2idx)
    print('Building vocab ... ...')
    vocab = Vocab(config, datasets.train_src)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # save pt(train, valid, test)
    save_data(datasets.train_src, datasets.train_tgt, vocab.word2idx, tokenizer, config.t_len, config.s_len,
              config.filename_trimmed_train)
    save_data(datasets.valid_src, datasets.valid_tgt, vocab.word2idx, tokenizer, config.t_len, config.s_len,
              config.filename_trimmed_valid)
    save_data(datasets.test_src, datasets.test_tgt, vocab.word2idx, tokenizer, config.t_len, config.s_len,
              config.filename_trimmed_test)


def test():
    config = Config()
    f = open('DATA/data/word2index.pkl', 'rb')
    word2idx = pickle.load(f)
    print(word2idx['<T>'])
    f = open('DATA/data/index2word.pkl', 'rb')
    idx2word = pickle.load(f)
    print(idx2word[105])

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test = torch.load(config.filename_trimmed_test)
    sen = index2sentence(np.array(test[200][0]), idx2word)
    print(test[200][0])
    summary = tokenizer.convert_ids_to_tokens(np.array(test[161][1]))
    print(sen)
    print(summary)


if __name__ == '__main__':
    # main()
    test()