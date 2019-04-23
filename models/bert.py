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

    def input_Norm(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x

    def forward(self, x):
        x = self.input_Norm(x)
        segments_tensors = torch.zeros_like(x)
        for i in range(self.s_len):
            m = x
            index = torch.tensor([i])
            if torch.cuda.is_available():
                index = index.cuda()
            m.index_fill_(1, index, self.mask_id)
            if self.fine_tune:
                predictions = self.model(m, segments_tensors)
            else:
                with torch.no_grad():
                    predictions = self.model(m, segments_tensors)
            predictions = torch.argmax(predictions[:, i], dim=-1)
            x[:, i] = predictions
        return x[:, 1:]
        # return x


class Bert_AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fine_tune = config.fine_tune
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.cls = config.cls
        self.sep = config.sep

    def input_Norm(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.cls).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x

    def forward(self, x):
        # print('x:', x.size())
        x = self.input_Norm(x)
        segments_tensors = torch.zeros_like(x)
        if self.fine_tune:
            h, _ = self.model(x, segments_tensors)
        else:
            with torch.no_grad():
                h, _ = self.model(x, segments_tensors)
        # (batch, len, hidden)
        # print(len(h))
        # print(h[-1].size())
        return h[-1][:, 1:, :]
        # return h[-1]
