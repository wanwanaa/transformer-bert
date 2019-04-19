class Config():
    def __init__(self):
        # data
        self.filename_train_src = 'DATA/raw_data/train.source'
        self.filename_train_tgt = 'DATA/raw_data/train.target'
        self.filename_valid_src = 'DATA/raw_data/valid.source'
        self.filename_valid_tgt = 'DATA/raw_data/valid.target'
        self.filename_test_src = 'DATA/raw_data/test.source'
        self.filename_test_tgt = 'DATA/raw_data/test.target'

        self.filename_trimmed_train = 'DATA/data/valid.pt'
        self.filename_trimmed_valid = 'DATA/data/valid.pt'
        self.filename_trimmed_test = 'DATA/data/test.pt'

        # vocab
        self.filename_idx2word = 'DATA/data/index2word.pkl'
        self.filename_word2idx = 'DATA/data/word2index.pkl'

        self.src_vocab_size = 523566
        self.tgt_vocab_size = 21128

        self.t_len = 70
        self.s_len = 30

        self.pad = 0
        self.unk = 100
        self.cls = 101
        self.sep = 102
        self.mask_id = 103
        self.bos = 104
        self.eos = 105

        # filename result
        #############################################
        self.filename_data = 'result/data/'
        self.filename_model = 'result/model/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #############################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        self.fine_tune = False
        self.model_size = 768
        self.n_head = 12
        self.d_ff = 2048
        self.warmup_steps = 4000
        self.ls = 0.1

        self.n_layer = 12
        self.dropout = 0.3
