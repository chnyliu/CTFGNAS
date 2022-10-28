import torch
from utils import data_loader
import time
import torch.nn.functional as F


class ModelManager(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.args, self.data = data_loader(self.args)
        self.log.info('{}: train/val/test:{}/{}/{}, features:{}, classes:{}.'.format(
            time.strftime('%Y%m%d-%H%M%S'), sum(self.data.train_mask), sum(self.data.val_mask),
            sum(self.data.test_mask), self.args.num_node_features, self.args.num_classes))

    def train(self, model):
        data = self.data
        model = model.cuda()
        criterion = F.nll_loss
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        val_loss, best_epoch, patience, test_acc = 99999.9999, 0, 0, 0
        try:
            for epoch in range(self.args.epochs):
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
                if out.size(1) != self.args.num_classes:
                    raise
                output = F.log_softmax(out, 1)
                loss_val = criterion(output[data.val_mask], data.y[data.val_mask].cuda())
                loss_val = loss_val.item()

                predict = output[data.test_mask.cuda()].max(1)[1].type_as(data.y[data.test_mask].cuda())
                correct = predict.eq(data.y[data.test_mask].cuda()).double()
                acc_test = correct.sum() / len(data.y[data.test_mask].cuda())
                acc_test = acc_test.item()

                if loss_val < val_loss:
                    val_loss = loss_val
                    best_epoch = epoch + 1
                    patience = 0
                    test_acc = acc_test

                    f = open("{}/temp.txt".format(self.args.save), 'r')
                    file = f.readlines()
                    f.close()
                    best_loss = float(file[0])
                    if val_loss < best_loss:
                        with open("{}/temp.txt".format(self.args.save), "w") as f:
                            f.write('{}'.format(val_loss))
                        torch.save(model, '{}/model.pth'.format(self.args.save))
                else:
                    patience += 1
                if patience >= 150:
                    break
                # if (epoch + 1) % 10 == 0 or epoch == 0:
                #     print('\t Epoch:{:03d}, train loss:{:.4f}, val loss:{:.4f}, test acc:{:.4f}.'.format(
                #         epoch + 1, loss_train, loss_val, test_acc))
            self.log.info('\t No.{:02d}: val loss:{:.4f}, test acc:{:.4f}, length:{}, best epoch:{:03d}.'.format(
                self.args.indi_no, val_loss, test_acc, model.num_layer, best_epoch))
            return round(val_loss, 4), round(test_acc, 4)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.log.info('\t No.{}: out of memory.'.format(self.args.indi_no))
                return round(val_loss, 4), round(test_acc, 4)
            else:
                raise
