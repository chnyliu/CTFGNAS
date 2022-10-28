import argparse
import random
import time
from utils import Log
import numpy as np
from population import Evolution_Trainer
import os


def get_args():
    parser = argparse.ArgumentParser(description='Implementation of CTFGNAS')
    # parameters for datasets
    parser.add_argument('--data', type=str, default='Cora', help='Cora, CiteSeer, PubMed, Texas, Wisconsin, or Actor')
    # parameters for EA
    parser.add_argument('--init_size', type=int, default=100)
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--max_gen', type=int, default=30)
    parser.add_argument('--max_len', type=int, default=8)
    parser.add_argument('--geno_len', type=int, default=5)
    parser.add_argument('--pc', type=float, default=0.9, help='Crossover probability')
    parser.add_argument('--pm', type=float, default=0.2, help='Mutation probability')
    parser.add_argument('--sn', type=float, default=30, help='surrogate number')
    parser.add_argument('--run_times', type=int, default=5)
    # parameters for training
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    # parameters for tuning
    parser.add_argument('--tune_genotype', type=np.array, default=np.random.random())
    parser.add_argument('--tune_epochs', type=int, default=300)
    parser.add_argument('--tune_times', type=int, default=50)
    # parameters for log
    parser.add_argument('--save', type=str, default='logs')
    args = parser.parse_args()
    return args


def main():
    beg_time, mid_time, end_time, predict_acc = [], [], [], []
    res_acc, res_loss, res_geno = [], [], []
    for index in range(args.run_times):
        args.hidden_dim = 64
        args.lr = 0.005
        args.wd = 0.0005
        args.dropout = 0.6
        args.time = time.strftime('%Y%m%d-%H%M%S')
        beg_time.append(args.time)
        args.save = 'logs/{}-{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), args.data, index+1)
        args.tune_genotype = np.random.random()
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        log = Log(args)
        args.seed = random.randint(0, 10000)
        with open("{}/temp.txt".format(args.save), "w") as f:
            f.write('{}'.format(99999.99))
        trainer = Evolution_Trainer(args, log)
        middle_time, top_loss, top_acc, top_geno, pre_acc_arc = trainer.train()
        mid_time.append(middle_time)
        res_acc.append(top_acc)
        res_loss.append(top_loss)
        res_geno.append(top_geno)
        predict_acc.append(pre_acc_arc)
        end_time.append(time.strftime('%Y%m%d-%H%M%S'))
        if index == args.run_times - 1:
            log.info('{}: Acc: {}, Mean: {:.4f}, Std: {:.4f}.'.format(time.strftime('%Y%m%d-%H%M%S'), res_acc, np.mean(res_acc), np.std(res_acc)))
            log.info('{}: Begin times: {}.'.format(time.strftime('%Y%m%d-%H%M%S'), beg_time))
            log.info('{}: Middle times: {}.'.format(time.strftime('%Y%m%d-%H%M%S'), mid_time))
            log.info('{}: End times: {}.'.format(time.strftime('%Y%m%d-%H%M%S'), end_time))
            log.info('{}: Predict Acc: {}'.format(time.strftime('%Y%m%d-%H%M%S'), predict_acc))


if __name__ == '__main__':
    args = get_args()
    main()
