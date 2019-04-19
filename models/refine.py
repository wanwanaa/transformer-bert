import torch
import torch.nn as nn
from utils import *


class Refine_model(nn.Module):
    def __init__(self, encoder, decoder, bert, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bert = bert
        self.linear_out = nn.Linear(config.model_size, config.tgt_vocab_size)
        self.t_len = config.t_len
        self.s_len = config.s_len
        self.pad = config.pad
        self.bos = config.bos

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
        # transformer
        enc_output = self.encoder(x, x_pos)
        dec_output = self.decoder(x, y, y_pos, enc_output)
        out = self.linear_out(dec_output)
        out = torch.nn.funvtional.softmax(out)
        out = torch.argmax(out, dim=-1)

        # refine
        refine = self.bert(out)
        refine = self.convert(refine)
        y_pos = torch.arange(1, self.s_len + 1).repeat(x.size(0), 1)
        y_mask = refine.eq(self.pad)
        y_pos = y_pos.masked_fill(y_mask, 0)
        dec_output = self.decoder(x, refine, y_pos, enc_output)
        final_out = self.linear_out(dec_output)
        return final_out

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
        # refine
        refine = self.bert(out[:, 1:])
        refine = self.convert(refine)
        y_mask = refine.eq(self.pad)
        y_pos = y_pos.masked_fill(y_mask, 0)
        dec_output = self.decoder(x, refine, y_pos, enc_output)
        dec_output = self.linear_out(dec_output)
        final_out = torch.nn.functional.softmax(dec_output, -1)
        outs = torch.argmax(final_out, dim=-1)
        return dec_output, outs
