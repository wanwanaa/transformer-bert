import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fine_tune = config.fine_tune
        self.s_len = config.s_len
        self.mask_id = config.mask_id
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')

    def forward(self, x):
        segments_tensors = torch.zeros_like(x)
        for i in range(self.s_len):
            m = x
            m.index_fill_(1, torch.tensor([i]), self.mask_id)
            if self.fine_tune:
                predictions = self.model(m, segments_tensors)
            else:
                with torch.no_grad():
                    predictions = self.model(m, segments_tensors)
            predictions = torch.argmax(predictions[:, i], dim=-1)
            x[:, i] = predictions
        return x


class Bert_AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fine_tune = config.fine_tune
        self.model = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, x):
        segments_tensors = torch.zeros_like(x)
        if self.fine_tune:
            h = self.model(x, segments_tensors)
        else:
            with torch.no_grad():
                h = self.model(x, segments_tensors)
        # (batch, len, hidden)
        return h[-1]



