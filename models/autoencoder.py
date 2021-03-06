import torch
import torch.nn as nn
from models.bert import Bert_AE
from models.transformer import Decoder


class AE(nn.Module):
    def __init__(self, encoder, decoder, bert, decoder_ae, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bert = bert
        self.decoder_ae = decoder_ae
        self.t_len = config.t_len
        self.s_len = config.s_len
        self.pad = config.pad
        self.bos = config.bos
        self.model_size = config.model_size
        self.linear_bert = nn.Linear(768, config.model_size)
        self.linear_out = nn.Linear(config.model_size, config.tgt_vocab_size)
        self.linear_ae = nn.Linear(config.model_size, config.tgt_vocab_size)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def forward(self, x, y):
        y = self.convert(y)
        x_pos = torch.arange(1, self.t_len + 1).repeat(x.size(0), 1)
        y_pos = torch.arange(1, self.s_len + 1).repeat(x.size(0), 1)
        if torch.cuda.is_available():
            x_pos = x_pos.cuda()
            y_pos = y_pos.cuda()
        x_mask = x.eq(self.pad)
        y_mask = y.eq(self.pad)
        x_pos = x_pos.masked_fill(x_mask, 0)
        y_pos = y_pos.masked_fill(y_mask, 0)
        # y = self.convert(y)
        # transformer
        enc_output = self.encoder(x, x_pos)
        dec_output = self.decoder(x, y, y_pos, enc_output)
        dec_output = self.linear_out(dec_output)
        out = torch.nn.functional.softmax(dec_output)
        out = torch.argmax(out, dim=-1)

        # print('out:', out.size())
        # bert hidden states
        h = self.bert(out)
        h = self.linear_bert(h)
        # print('h:', h.size())
        output = self.decoder_ae(out, y, y_pos, h)
        final_out = self.linear_ae(output)
        return dec_output, final_out

    def sample(self, x):
        x_pos = torch.arange(1, self.t_len + 1).repeat(x.size(0), 1)
        y_pos = torch.arange(1, self.s_len + 1).repeat(x.size(0), 1)
        if torch.cuda.is_available():
            x_pos = x_pos.cuda()
            y_pos = y_pos.cuda()
        x_mask = x.eq(self.pad)
        x_pos = x_pos.masked_fill(x_mask, 0)
        # encoder
        enc_output = self.encoder(x, x_pos)

        # <start> connect to decoding input at each step
        start = torch.ones(x.size(0)) * self.bos
        start = start.unsqueeze(1)
        if torch.cuda.is_available():
            start = start.type(torch.cuda.LongTensor)
        else:
            start = start.type(torch.LongTensor)
        # the first <start>
        out = torch.ones(x.size(0)) * self.bos
        out = out.unsqueeze(1)
        dec_output = None
        # decoder
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            dec_output = self.decoder(x, out, y_pos[:, :i + 1], enc_output)
            dec_output = self.linear_out(dec_output)  # (batch, len, vocab_size)
            gen = torch.nn.functional.softmax(dec_output, -1)
            gen = torch.argmax(gen, dim=-1)  # (batch, len) eg. 1, 2, 3
            # print(gen.size())
            out = torch.cat((start, gen), dim=1)  # (batch, len+1) eg. <start>, 1, 2, 3
        outs = out[:, 1:]
        # bert autoencoder
        h = self.bert(outs)
        h = self.linear_bert(h)

        # the first <start>
        out = torch.ones(x.size(0)) * self.bos
        out = out.unsqueeze(1)
        # decoder
        final_output = None
        for i in range(self.s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            final_output = self.decoder(outs, out, y_pos[:, :i + 1], h)
            final_output = self.linear_ae(final_output)  # (batch, len, vocab_size)
            gen = torch.nn.functional.softmax(final_output, -1)
            gen = torch.argmax(gen, dim=-1)  # (batch, len) eg. 1, 2, 3
            out = torch.cat((start, gen), dim=1)  # (batch, len+1) eg. <start>, 1, 2, 3
        return dec_output, final_output, out[:, 1:]
