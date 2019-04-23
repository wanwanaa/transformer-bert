from utils import *
from models import *
import argparse
from tqdm import tqdm


def valid(epoch, config, model, loss_func):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    valid_loader = data_load(config.filename_trimmed_valid, args.batch_size, False)
    all_loss = 0
    num = 0
    for step, batch in enumerate(tqdm(valid_loader)):
        # num += 1
        x, y = batch
        num += y.ne(config.pad).sum().item()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            decoder_out, final_out,  _ = model.sample(x)
        loss = loss_func(decoder_out, y) + loss_func(final_out, y)
        all_loss += loss.item()
        #########################
        if step == 2:
            break
        #########################
    print('epoch:', epoch, '|valid_loss: %.4f' % (all_loss / num))
    return all_loss / num


def test(epoch, config, model, loss_func, tokenizer):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # data
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    all_loss = 0
    num = 0
    r = []
    for step, batch in enumerate(tqdm(test_loader)):
        # num += 1
        x, y = batch
        num += y.ne(config.pad).sum().item()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            decoder_out, final_out, out = model.sample(x)
        loss = loss_func(decoder_out, y) + loss_func(final_out, y)
        all_loss += loss.item()

        out = out.cpu().numpy()
        for i in range(out.shape[0]):
            t = []
            for c in list(out[i]):
                if c == 102:
                    break
                t.append(c)
            if len(t) == 0:
                sen = []
                sen.append('[UNK]')
            else:
                sen = tokenizer.convert_ids_to_tokens(t)
            r.append(' '.join(sen))
        #########################
        if step == 2:
            break
        #########################

    print('epoch:', epoch, '|test_loss: %.4f' % (all_loss / num))

    # write result
    filename_data = config.filename_data + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # rouge
    score = rouge_score(config.filename_gold, filename_data)

    # write rouge
    write_rouge(config.filename_rouge, score, epoch)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])

    return score['rouge-2']['f'], all_loss / num


def train(args, config, model):
    max_sorce = 0.0
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-9)
    optim = Optim(optimizer, config)
    # KLDivLoss
    loss_func = LabelSmoothing(config)

    train_loader = data_load(config.filename_trimmed_train, args.batch_size, True)

    if args.checkpoint != 0:
        model.load_state_dict(torch.load(config.filename_model + 'model_' + str(args.checkpoint) + '.pkl'))
        args.checkpoint += 1

    for e in range(args.checkpoint, args.epoch):
        model.train()
        all_loss = 0
        num = 0
        for step, batch in enumerate(tqdm(train_loader)):
            # num += 1
            x, y = batch
            num += y.ne(config.pad).sum().item()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            decout_out, final_out = model(x, y)

            loss = loss_func(decout_out, y) + loss_func(final_out, y)
            all_loss += loss.item()
            #########################
            if step == 2:
                break
            #########################
            if step % 200 == 0:
                # print(final_out[-1].size())
                decout_out = torch.nn.functional.softmax(decout_out[-1], dim=-1)
                # print(decout_out.size())
                decout_out = torch.argmax(decout_out, dim=-1)
                print(decout_out.size())
                final_out = torch.nn.functional.softmax(final_out[-1], dim=-1)
                final_out = torch.argmax(final_out, dim=-1)
                if torch.cuda.is_available():
                    decout_out = decout_out.cpu().numpy()
                    final_out = final_out.cpu().numpy()
                    y = y[-1].cpu().numpy()
                else:
                    decout_out = decout_out.numpy()
                    final_out = final_out.numpy()
                    y = y[-1].numpy()
                # print(decout_out)
                decout_out = tokenizer.convert_ids_to_tokens(list(decout_out))
                final_out = tokenizer.convert_ids_to_tokens(list(final_out))
                y = tokenizer.convert_ids_to_tokens(list(y))
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % (loss.item() / num))
                print(''.join(decout_out))
                print(''.join(final_out))
                print(''.join(y))
            # if step != 0 and step % 500 == 0:
            #     test(e, config, model, loss_func, tokenizer)
            if step != 0 and step % 5000 == 0:
                filename = config.filename_model + 'model_' + str(step) + '.pkl'
                save_model(model, filename)

            optim.zero_grad()
            loss.backward()
            optim.updata()
        # filename = config.filename_model + 'model_' + str(e) + '.pkl'
        # save_model(model, filename)
        # train loss
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)

        # valid
        # valid(e, config, model, loss_func)

        # test
        score, _ = test(e, config, model, loss_func, tokenizer)
        if score > max_sorce:
            max_sorce = score
            filename = config.filename_model + 'model.pkl'
            save_model(model, filename)


if __name__ == '__main__':
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size for train')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='number of training epochs')
    parser.add_argument('--n_layer', '-n', type=int, default=6, help='number of encoder layers')
    parser.add_argument('-seed', '-s', type=int, default=123, help="Random seed")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")
    parser.add_argument('--checkpoint', '-c', type=int, default=0, help="load model")
    args = parser.parse_args()

    # ########test##########
    args.batch_size = 1
    # ########test##########

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_layer:
        config.n_layer = args.n_layer

    # seed
    torch.manual_seed(args.seed)

    # rouge initalization
    open(config.filename_rouge, 'w')

    # model = build_refine(config)
    model = build_autoencoder(config)
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train(args, config, model)

