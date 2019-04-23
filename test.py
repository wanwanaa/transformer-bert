import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer
from models.loss import LabelSmoothing
from models.build_model import *
from utils import *
from train import test

if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    loss_func = LabelSmoothing(config)
    filename = 'result/model/model_0.pkl'
    model = load_model(config, filename)
    test(0, config, model, loss_func, tokenizer)