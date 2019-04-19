import torch
import torch.nn as nn
from models.layer import EncoderLayer, DecoderLayer
from models.transformer_helper import *


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.src_vocab_size = config.src_vocab_size
        self.model_size = config.model_size
        self.pad = config.pad

        # word embedding
        self.embedding = nn.Embedding(self.src_vocab_size, self.model_size)

        # positional Encoding
        self.position_enc = nn.Embedding.from_pretrained(
            positional_encoding(config.t_len+1, config.model_size, config.pad), freeze=True
        )

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(self.n_layer)
        ])
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, x, pos):
        # mask
        non_pad_mask = get_non_pad_mask(x, self.pad) # (batch, len, 1)
        attn_mask = get_attn_pad_mask(x, self.pad) # (batch, len, len)

        enc_output = self.embedding(x) + self.position_enc(pos)
        enc_output = self.layer_norm(enc_output)
        for layer in self.encoder_stack:
            enc_output, enc_attn = layer(enc_output, non_pad_mask, attn_mask)

        return enc_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad = config.pad

        self.embedding = nn.Embedding(config.tgt_vocab_size, config.model_size)

        self.position_dec = nn.Embedding.from_pretrained(
            positional_encoding(config.s_len+1, config.model_size, config.pad), freeze=True
        )

        self.decoder_stack = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layer)
        ])
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, x, y, pos, enc_output):
        no_pad_mask = get_non_pad_mask(y, self.pad)
        attn_mask = get_dec_mask(y)
        pad_mask = get_pad_mask(y, y, self.pad)
        attn_self_mask = (pad_mask + attn_mask).gt(0)
        enc_dec_attn_mask = get_pad_mask(x, y, self.pad)

        dec_output = self.embedding(y) + self.position_dec(pos)
        enc_output = self.layer_norm(enc_output)
        for layer in self.decoder_stack:
            dec_output, _, _ = layer(dec_output, enc_output, no_pad_mask, attn_self_mask, enc_dec_attn_mask)

        return dec_output