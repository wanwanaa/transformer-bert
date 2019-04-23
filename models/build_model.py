from models import *
from utils import *
from models.transformer import *
from models.refine import *
from models.autoencoder import *


def build_refine(config):
    encoder = Encoder(config)
    decoder = Decoder(config)
    bert = Bert(config)
    model = Refine_model(encoder, decoder, bert, config)
    return model


# # autoencoder
# def build_autoencoder(config):
#     encoder = Encoder(config)
#     decoder = Decoder(config)
#     bert = Bert_AE(config)
#     decoder_ae = Decoder(config)
#     model = AE(encoder, decoder, bert, decoder_ae, config)
#     return model


# autoencoder
def build_autoencoder(config):
    encoder = Encoder(config)
    decoder = Decoder(config)
    decoder_ae = Decoder(config)
    bert = Bert_AE(config)
    model = AE(encoder, decoder, bert, decoder_ae, config)
    return model


def load_model(config, filename):
    model = build_autoencoder(config)
    # model = build_refine(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)
    print('model save at ', filename)