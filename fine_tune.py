import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import torch
from network import Network
from utils import data_loader
import time
import torch.nn.functional as F


def hyperopt_train(params):
    params.hidden_dim = params.hidden_size
    params, data = data_loader(params)
    model = Network(params, None, params.genotype_simple)
    model = model.cuda()

    criterion = F.nll_loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    val_loss, best_epoch, patience, test_acc = 99999.9999, 0, 0, 0
    f = open("{}/temp.txt".format(params.save), 'r')
    file = f.readlines()
    f.close()
    best_loss = float(file[0])
    try:
        for epoch in range(params.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x.cuda(), data.edge_index.cuda())
            out = F.log_softmax(out, 1)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].cuda())
            loss.backward()
            loss_train = loss.item()
            optimizer.step()

            model.eval()
            out = model(data.x.cuda(), data.edge_index.cuda())
            output = F.log_softmax(out, 1)
            loss_val = criterion(output[data.val_mask], data.y[data.val_mask].cuda())
            loss_val = loss_val.item()
            predit = output[data.test_mask.cuda()].max(1)[1].type_as(data.y[data.test_mask].cuda())
            correct = predit.eq(data.y[data.test_mask].cuda()).double()
            acc_test = correct.sum() / len(data.y[data.test_mask].cuda())
            acc_test = acc_test.item()

            if loss_val < val_loss:
                val_loss = loss_val
                patience = 0
                test_acc = acc_test
                if val_loss < best_loss:
                    with open("{}/temp.txt".format(params.save), "w") as f:
                        f.write('{}'.format(val_loss))
                    torch.save(model, '{}/model.pth'.format(params.save))
            else:
                patience += 1
            if patience >= 200:
                break
            # if ((epoch + 1) % 10 == 0 or epoch == 0):
            #     print('\t Epoch:{:03d}, train loss:{:.4f}, val loss:{:.4f}.'.
            #           format(epoch + 1, loss_train, loss_val))
        logs.info('\t Val loss:{:.4f}, test acc:{:.4f}, hidden size:{}, learning rate:{}, weight_decay:{}, dropout:{}'.
                  format(val_loss, test_acc, params.hidden_dim, params.learning_rate, params.weight_decay,
                         params.dropout))
        return round(val_loss, 4), round(test_acc, 4)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            logs.info('\t out of memory.')
            return round(val_loss, 4), round(test_acc, 4)
        else:
            raise


class Params(object):
    def __init__(self):
        super(Params, self).__init__()


search_space = {'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256, 512]),
                'learning_rate': hp.choice("lr", [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]),
                'weight_decay': hp.choice("wr", [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]),
                'dropout': hp.choice('in_dropout', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                }


def generate_params(params):
    train_params = Params()
    for key, value in params.items():
        setattr(train_params, key, value)
    train_params.data = args.data

    train_params.epochs = args.tune_epochs
    train_params.seed = args.seed
    train_params.save = args.save
    train_params.genotype_simple = args.tune_genotype
    train_params.geno_len = args.geno_len
    return train_params


def objective(params):
    train_params = generate_params(params)
    val_loss, test_acc = hyperopt_train(train_params)
    return {'loss': val_loss,
            'acc': test_acc,
            'status': STATUS_OK
            }


def run_fine_tune(arg, log, individual, top1_loss, top1_acc, top1_geno):
    global args, logs
    args = arg
    logs = log
    log.info('{}: Tune. val loss:{}, acc:{}, genotype:{}, .'.
             format(time.strftime('%Y%m%d-%H%M%S'), individual.val_loss, individual.test_acc, individual.genoBin_sim))
    trials = Trials()
    best = fmin(fn=objective,
                space=search_space,
                algo=partial(tpe.suggest, n_startup_jobs=10),
                max_evals=args.tune_times,
                trials=trials)
    space = hyperopt.space_eval(search_space, best)
    val_loss, test_acc = 99999.99, 0
    for d in trials.results:
        if d['loss'] < val_loss:
            val_loss = d['loss']
            test_acc = d['acc']
    if val_loss < top1_loss:
        log.info('\t {}: Best space is {}'.format(time.strftime('%Y%m%d-%H%M%S'), space))
        logs.info('\t {}: Fine tune. Val loss:{}, test acc:{}, genotype:{}'.
                  format(time.strftime('%Y%m%d-%H%M%S'), val_loss, test_acc, args.tune_genotype))
        args.hidden_dim, args.lr, args.wd, args.dropout = space['hidden_size'], space['learning_rate'], space[
            'weight_decay'], space['dropout']
        return args, val_loss, test_acc, args.tune_genotype
    else:
        logs.info('\t {}: No contribution.'.format(time.strftime('%Y%m%d-%H%M%S')))
        return args, top1_loss, top1_acc, top1_geno
